import torch

import sys
sys.path.append('../')
from functions.functions import STFT1024, ISTFT1024, predict_5class_pilot, count_SIR, get_signal_and_interference
from functions.iFICA import iFICA
from classes.classifiers import MainSpeakerIntensity5Classifier

import numpy as np
import torch

class HybridSystemForExtraction():
    """
    Hybridní systém pro extrakci hlavního mluvčího z vícekanálového směsového signálu.

    Tento systém využívá neuronovou síť k odhadu intenzity hlavního mluvčího a následně 
    algoritmus iFICA k separaci signálů s postranní informací z odhadnutého pilotního signálu.

    Parametry:
    - model_path: cesta k souboru s uloženými vahami neuronového modelu
    - feature_extraction_params: list parametrů pro extrakci rysů, které budou použity v modelu, obsahuje tuple hodnoty ve formátu (jméno typu parametru, (index jednoho mikrofonu, index druhého mikrofonu))
    """
    def __init__(self, model_path, feature_extraction_params):
        self.model = MainSpeakerIntensity5Classifier(number_of_features=len(feature_extraction_params))
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        self.feature_extraction_params = feature_extraction_params

    def extract_main_speaker(self, mixture):
        """
        Extrahuje signál hlavního mluvčího z vícekanálového směsného signálu.

        Parametry:
        - mixture: vstupní směsný signál v časové oblasti pro více mikrofonů

        Návratová hodnota:
        - extracted_s: extrahovaný signál hlavního mluvčího v časové oblasti
        """
        self.model.eval()

        X = STFT1024(mixture.T).transpose(0,2,1)
        pilot = predict_5class_pilot(model=self.model, mixture=mixture, feature_parameters=self.feature_extraction_params)

        number_of_channels = X.shape[0]
        [_, _, shat, _] = iFICA(X, np.ones((number_of_channels,1,X.shape[2])), pilot);
        shat_reshaped = np.transpose(shat, (1, 0, 2))
        extracted_s = ISTFT1024(shat_reshaped)

        return extracted_s

    def evaluate_extraction_of_main_speaker(self, mixture, soi, interference):
        """
        Vyhodnocuje účinnost extrakce hlavního mluvčího pomocí metriky SIR.

        Parametry:
        - mixture: vstupní směsný signál v časové oblasti
        - soi: původní signál hlavního mluvčího v časové oblasti
        - interference: interferenční signál v časové oblasti

        Návratová hodnota:
        - SIR_original: původní SIR před extrakcí
        - SIR_final: výsledný SIR po extrakci
        - SIR_improvement: zlepšení SIR po extrakci
        """
        self.model.eval()

        SIR_original, _ = count_SIR(soi, interference)

        X = STFT1024(mixture.T).transpose(0,2,1)
        pilot = predict_5class_pilot(model=self.model, mixture=mixture, feature_parameters=self.feature_extraction_params)

        number_of_channels = X.shape[0]
        d, N, K = X.shape
        SOI = STFT1024(soi.T).transpose(0,2,1)
        INT = STFT1024(interference.T).transpose(0,2,1)
        
        [w, _, _, _] = iFICA(X, np.ones((number_of_channels,1,X.shape[2])), pilot);

        x_soi, x_int = get_signal_and_interference(w, SOI, INT)
        
        SIR_final, _ = count_SIR(x_soi, x_int)
        SIR_improvement = SIR_final - SIR_original

        return SIR_original, SIR_final, SIR_improvement

class HybridSystem_IPD17(HybridSystemForExtraction):
    """ Hybridní systém využívající pouze IPD mezi mikrofony 1 a 7. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ipd", (1, 7))])


class HybridSystem_IPD35(HybridSystemForExtraction):
    """ Hybridní systém využívající pouze IPD mezi mikrofony 3 a 5. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ipd", (3, 5))])


class HybridSystem_IPD14_IPD17(HybridSystemForExtraction):
    """ Hybridní systém využívající IPD mezi mikrofony 1-4 a 1-7. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ipd", (1, 4)), ("ipd", (1, 7))])


class HybridSystem_IPD14_IPD17_IPD35(HybridSystemForExtraction):
    """ Hybridní systém využívající IPD mezi mikrofony 1-4, 1-7 a 3-5. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ipd", (1, 4)), ("ipd", (1, 7)), ("ipd", (3, 5))])


class HybridSystem_ILD17(HybridSystemForExtraction):
    """ Hybridní systém využívající pouze ILD mezi mikrofony 1 a 7. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ild", (1, 7))])


class HybridSystem_ILD35(HybridSystemForExtraction):
    """ Hybridní systém využívající pouze ILD mezi mikrofony 3 a 5. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ild", (3, 5))])


class HybridSystem_ILD14_ILD17(HybridSystemForExtraction):
    """ Hybridní systém využívající ILD mezi mikrofony 1-4 a 1-7. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ild", (1, 4)), ("ild", (1, 7))])


class HybridSystem_ILD14_ILD17_ILD35(HybridSystemForExtraction):
    """ Hybridní systém využívající ILD mezi mikrofony 1-4, 1-7 a 3-5. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ild", (1, 4)), ("ild", (1, 7)), ("ild", (3, 5))])


class HybridSystem_IPD17_ILD17(HybridSystemForExtraction):
    """ Hybridní systém kombinující IPD a ILD mezi mikrofony 1 a 7. """
    def __init__(self, model_path):
        super().__init__(model_path=model_path, feature_extraction_params=[("ipd", (1, 7)), ("ild", (1, 7))])