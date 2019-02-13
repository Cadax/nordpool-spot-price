# Nordpoolin tunnittaisen spot-hinnan ennustaminen Suomessa

* Ladattu dataa: Spot-hinnat tunnin välein, lämpötilamittaukset, sähkönsiirto, sähkönkulutus, tuulivoima, ydinvoima, vesivoima, päästöoikeuksien hinnat
* Yhdistetty yhteen dataframeen: Spot-hinnat, lämpötilamittaukset, sähkönsiirto, sähkönkulutus, tuulivoima, ydinvoima, vesivoima


Ennustustuloksia:

| Menetelmä | RMSE | MAE |
| --- | --- | --- |
| LSTM verkko pelkällä Spot-hinnan historiatiedolla | 9,86 | 7,79 |
| CNN-LSTM verkko pelkällä Spot-hinnan historiatiedolla | 18,04 | 13,60 |
| ConvLSTM verkko pelkällä Spot-hinnan historiatiedolla | - | - |
| FBProphet additiivinen malli pelkällä Spot-hinnan historiatiedolla | 10,46 | 6,81 |   
| LSTM verkko usealla muuttujalla | - | - |
| CNN-LSTM verkko usealla muuttujalla | - | - |
| ConvLSTM verkko usealla muuttujalla | - | - |
