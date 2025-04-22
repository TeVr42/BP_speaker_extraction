import numpy as np
from scipy.signal import stft, istft
import torch
from classes.classifiers import MainSpeakerIntensity5Classifier

import sys
sys.path.append('./')
from functions.iFICA import iFICA

def count_SIR(signal, interference):
    """
    Výpočet poměru signálu k interferenci (Signal-to-Interference Ratio, SIR).
    
    Parametry:
    - signal: numpy pole obsahující požadovaný signál
    - interference: numpy pole obsahující interferenční složku
    
    Návratová hodnota:
    - mean_SIR: průměrný SIR v dB
    - SIRs: pole obsahující SIR hodnoty pro jednotlivé vzorky
    """
    signal_energy = np.sum(signal**2)
    interference_energy = np.sum(interference**2)

    SIRs = 10 * np.log10(signal_energy / interference_energy)
    mean_SIR = np.mean(SIRs)

    return mean_SIR, SIRs

def STFT1024(signal):
    """
    Výpočet krátkodobé Fourierovy transformace (STFT) s pevně definovanými parametry.
    
    Parametry:
    - signal: vstupní časový signál
    
    Návratová hodnota:
    - X: spektrální reprezentace signálu
    """
    NFFT = 1024
    SHIFT = 128
    window = np.hamming(NFFT)
    _, _, X = stft(signal, nperseg=NFFT, noverlap=NFFT - SHIFT, nfft=NFFT, window=window, return_onesided=True)
    return X

def ISTFT1024(signal):
    """
    Výpočet inverzní krátkodobé Fourierovy transformace (ISTFT).
    
    Parametry:
    - signal: spektrální reprezentace signálu
    
    Návratová hodnota:
    - x: rekonstruovaný časový signál
    """
    NFFT = 1024
    SHIFT = 128
    window = np.hamming(NFFT)
    _, x = istft(signal, nperseg=NFFT, noverlap=NFFT - SHIFT, nfft=NFFT, window=window)
    return x

def get_signal_and_interference(w, SOI, INT):
    """
    Rekonstrukce odděleného signálu a interference na základě separační matice w.
    
    Parametry:
    - w: separační matice
    - SOI: matice obsahující signál zájmu (Source of Interest)
    - INT: matice obsahující interferenční složky
    
    Návratová hodnota:
    - x_soi: rekonstruovaný signál zájmu
    - x_int: rekonstruovaná interference
    """
    w_conj_transpose = np.conj(np.transpose(w, (0, 2, 1)))

    shat_soi = np.matmul(w_conj_transpose, SOI.transpose(2,0,1))
    shat_soi_reshaped = np.transpose(shat_soi, (1, 0, 2))
    x_soi = ISTFT1024(shat_soi_reshaped)

    shat_int = np.matmul(w_conj_transpose, INT.transpose(2,0,1))
    shat_int_reshaped = np.transpose(shat_int, (1, 0, 2))
    x_int = ISTFT1024(shat_int_reshaped)

    return x_soi, x_int

def calculate_IPD(signal, mic1_nr, mic2_nr):
    """
    Výpočet fázového rozdílu mezi dvěma mikrofony (Interaural Phase Difference - IPD).
    
    Parametry:
    - signal: časový signál
    - mic1_nr, mic2_nr: indexy mikrofonů
    
    Návratová hodnota:
    - ipd: tensor obsahující IPD hodnoty
    """
    X = STFT1024(signal.T).transpose(0,2,1)

    mic1 = X[mic1_nr, :, :]
    mic2 = X[mic2_nr, :, :]

    ipd = np.angle(mic2 / (mic1 + 1e-8))

    return torch.tensor(ipd, dtype=torch.float32)

def calculate_ILD(signal, mic1_nr, mic2_nr):
    """
    Výpočet rozdílu intenzity mezi dvěma mikrofony (Interaural Level Difference - ILD).
    
    Parametry:
    - signal: časový signál
    - mic1_nr, mic2_nr: indexy mikrofonů
    
    Návratová hodnota:
    - ild: tensor obsahující ILD hodnoty
    """
    X = STFT1024(signal.T).transpose(0,2,1)

    mic1 = X[mic1_nr, :, :]
    mic2 = X[mic2_nr, :, :]

    mag1 = np.abs(mic1)
    mag2 = np.abs(mic2)

    ild = 20 * np.log10(mag2 / (mag1 + 1e-8) + 1e-8)

    return torch.tensor(ild, dtype=torch.float32)


def predict_5class_pilot(model: MainSpeakerIntensity5Classifier, mixture, feature_parameters):
    """
    Predikce intenzity hlavního mluvčího na základě extrahovaných akustických rysů.
    
    Parametry:
    - model: neuronová síť pro klasifikaci intenzity
    - mixture: směs audio signálů
    - feature_parameters: parametry určující, které akustické rysy budou extrahovány
    
    Návratová hodnota:
    - pilot: pole obsahující predikovanou intenzitu v rozmezí [0.0, 1.0]
    """
    model.eval()
    
    extracted_fetures = []
    for feature_extraction_param in feature_parameters:
        if feature_extraction_param[0] == "ipd":
            feature = calculate_IPD(mixture, feature_extraction_param[1][0], feature_extraction_param[1][1])
        elif feature_extraction_param[0] == "ild":
            feature = calculate_ILD(mixture, feature_extraction_param[1][0], feature_extraction_param[1][1])
        else:
            raise ValueError("Unknown feature extraction parameter")
        extracted_fetures.append(feature)
    
    with torch.no_grad():
        predictions = []
        n_frames, n_freqs = extracted_fetures[0].shape
        padded_features = []

        # rozsireni o padding
        for feature in extracted_fetures:
            padded_feature = torch.full((n_frames + 4, n_freqs), 0.0)
            padded_feature[2:-2, :] = feature
            padded_features.append(padded_feature)

        for i in range(n_frames):
            context_list = []
            # pripraveni kontextu pres jednotlive kanaly
            for j in range(len(padded_features)):
                context_list.append(padded_features[j][i:i + 5, :])
            
            # predikce online
            context = torch.stack(context_list, dim=0) if len(context_list) > 1 else context_list[0]
            output = model(context.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predictions.append(int(predicted))

    mapping = {0: 0.0, 1: 0.25, 2:  0.5, 3: 0.75, 4: 1.0}
    pilot = [mapping[pred] for pred in predictions]
    return np.array(pilot)


def calculate_SIR_for_frames(S, Y):
    """
    Výpočet hodnot SIR (Signal-to-Interference Ratio) pro jednotlivé časové rámce.

    Parametry:
    - S: matice obsahující spektrální reprezentaci signálu zájmu (SOI)
    - Y: matice obsahující spektrální reprezentaci interferencí

    Návratová hodnota:
    - sir_values: pole obsahující SIR hodnoty pro jednotlivé časové rámce
    """
    sir_values = np.zeros(S.shape[2])
    
    for t in range(S.shape[2]):
        signal_power = np.sum(np.abs(S[:, :, t])**2)
        interference_power = np.sum(np.abs(Y[:, :, t])**2)
        sir_values[t] = 10 * np.log10(signal_power / (interference_power + 0.0001) + 0.0001)
    
    return sir_values

def create_ideal_pilot_from_SIR(sir_values):
    """
    Mapování ideálního pilotního signálu na základě hodnot SIR.

    Parametry:
    - sir_values: pole obsahující hodnoty SIR pro jednotlivé časové rámce

    Návratová hodnota:
    - pilot: pole obsahující hodnoty v rozmezí [0.0, 1.0] odpovídající diskretizované úrovni SIR
    """
    pilot = np.zeros_like(sir_values)
    
    pilot[sir_values >= 4.5] = 1.0
    pilot[(sir_values >= 3.0) & (sir_values < 4.5)] = 0.75
    pilot[(sir_values >= 1.5) & (sir_values < 3.0)] = 0.5
    pilot[(sir_values >= 0.0) & (sir_values < 1.5)] = 0.25
    pilot[sir_values < 0.0] = 0.0
    
    return pilot


def get_SIR_of_iFICA_result(nr_of_mics, K, X, SOI, INT, pilot):
    """
    Výpočet výsledného SIR po aplikaci algoritmu iFICA.

    Parametry:
    - nr_of_mics: počet mikrofonů v systému
    - K: počet nezávislých komponent
    - X: spektrální reprezentace vstupního signálu
    - SOI: spektrální reprezentace signálu zájmu
    - INT: spektrální reprezentace interferencí
    - pilot: pilotní signál pro algoritmus iFICA

    Návratová hodnota:
    - SIR_final: výsledná hodnota SIR po separaci
    """
    [w, _, _, _] = iFICA(X, np.ones((nr_of_mics,1,K)), pilot);
    x_soi, x_int = get_signal_and_interference(w, SOI, INT)
    SIR_final, _ = count_SIR(x_soi, x_int)
    return SIR_final