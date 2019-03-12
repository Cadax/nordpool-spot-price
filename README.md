# Nordpoolin tunnittaisen spot-hinnan ennustaminen Suomessa

## Muuttujat

* Yhdistetty yhteen dataframeen: Spot-hinnat, lämpötilamittaukset(HEL,JKL,ROV), sähkönsiirto, sähkönkulutus, tuulivoima, ydinvoima, vesivoima, kivihiilen hinta, öljyn (BRENT) hinta
* Puuttuu: Ruotsin ydinvoima ja tuulivoima. Päästöoikeuksien hinnat.

## Ennustustulokset

| Menetelmä | RMSE | MAE |
| --- | --- | --- |
| Random Forest (testi datasetistä viimeiset 10%) | 8,18 | 5,67 |
| Random Forest (pelkkä 2017) | 7,94 | 4,86 |
| Random Forest (uusi datasetti) | 6,70 | 4,56 |
| LSTM verkko pelkällä Spot-hinnan historiatiedolla (testi: tammikuu 2019) (timestep 1h) | 9,86 | 7,79 |
| CNN-LSTM verkko pelkällä Spot-hinnan historiatiedolla (testi: tammikuu 2019) | 18,04 | 13,60 |
| ConvLSTM verkko pelkällä Spot-hinnan historiatiedolla | - | - |
| FBProphet additiivinen malli pelkällä Spot-hinnan historiatiedolla (testi: tammikuu 2019) | 10,46 | 6,81 |   
| LSTM verkko usealla muuttujalla (testi: vuosi 2018) (timestep 1h) | 6,18 | 3,61 |
| LSTM Encoder Decoder usealla muuttujalla (timestep 168h) | 11,24 | 8,14 |
| CNN-LSTM verkko usealla muuttujalla | - | - |
| ConvLSTM verkko usealla muuttujalla | - | - |