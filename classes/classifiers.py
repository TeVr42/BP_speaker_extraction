import torch.nn as nn

class MainSpeakerIntensity5Classifier(nn.Module):
    """
    Neuronová síť pro klasifikaci intenzity hlavního mluvčího do 5 tříd.
    
    Síť využívá konvoluční vrstvy k extrakci relevantních rysů z akustických dat
    a plně propojené vrstvy k následné klasifikaci do jedné z pěti úrovní intenzity.

    Parametry:
    - number_of_features: počet vstupních akustických rysů (implicitně 1)
    """
    def __init__(self, number_of_features=1):
        super(MainSpeakerIntensity5Classifier, self).__init__()

        self.number_of_features = number_of_features

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.number_of_features, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 128, 256),
            nn.ReLU(),

            nn.Linear(256, 5)
        )

    def forward(self, x):
        """
        Průchod vstupních dat neuronovou sítí.

        Parametry:
        - x: vstupní tensor s extrahovanými rysy

        Návratová hodnota:
        - Výstupní tensor s pravděpodobnostmi pro jednotlivé třídy intenzity
        """

        if self.number_of_features == 1:
            x = x.unsqueeze(1)
       
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc_layers(x)

        return x
