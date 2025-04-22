import os
import torch
from torch.utils.data import Dataset
import glob

class FeatureDataset(Dataset):
    """
    Vlastní dataset pro načítání extrahovaných rysů (features) a odpovídajících výstupních štítků (pilotních hodnot).

    Tento dataset slouží k trénování modelu, který předpovídá pilotní signál na základě extrahovaných rysů.

    Parametry:
    - input_dir: cesta k adresáři obsahujícímu vstupní feature soubory (např. IPD, ILD).
    - output_dir: cesta k adresáři obsahujícímu výstupní štítky (pilotní signály).
    - feature_names: seznam názvů rysů, které se mají načítat (např. ["ipd17_", "ild35_"]).
    - default_padding_value: hodnota, kterou se doplní okraje vstupních dat při kontextovém výběru (defaultně 0.0).
    - n_frames: počet snímků v každém souboru, pokud není zadán, odvodí se z prvního souboru.
    """
    def __init__(self, input_dir, output_dir, feature_names, default_padding_value=0.0, n_frames=None):
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        self.input_files = [
            sorted(glob.glob(os.path.join(input_dir, f'{feature_name}*.pt')))
            for feature_name in feature_names
        ]
        self.output_files = sorted(glob.glob(os.path.join(output_dir, 'pilot*.pt')))
        self.default_padding_value = default_padding_value
        self.n_frames = n_frames

        if n_frames is None:
            self.n_frames = torch.load(self.input_files[0][0], weights_only=True).shape[0]

    def __len__(self):
        """
        Vrací celkový počet trénovacích příkladů v datasetu.
        Každý snímek v rámci všech souborů se počítá jako samostatný příklad.
        """
        return self.n_frames * len(self.input_files[0])

    def __getitem__(self, index):
        """
        Načítá konkrétní trénovací příklad (kontextové okno rysů a odpovídající štítek).

        Parametry:
        - index: index požadovaného vzorku.

        Návratová hodnota:
        - context: tensor obsahující kontextové okno rysů kolem daného snímku.
        - label: diskrétní hodnota odpovídající pilotnímu signálu (0 až 4).
        """
        file_idx = index // self.n_frames
        frame_idx = index % self.n_frames

        feature_data = [torch.load(files[file_idx], weights_only=True) for files in self.input_files]
        pilot_data = torch.load(self.output_files[file_idx], weights_only=True)

        n_frames, n_freqs = feature_data[0].shape

        padded_features = []
        for data in feature_data:
            padded = torch.full((n_frames + 4, n_freqs), self.default_padding_value)
            padded[2:-2, :] = data
            padded_features.append(padded[frame_idx:frame_idx + 5, :])
        
        context = torch.stack(padded_features, dim=0) if len(padded_features) > 1 else padded_features[0]
        
        mapping = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
        label = mapping[pilot_data[frame_idx].item()]

        return context, label
