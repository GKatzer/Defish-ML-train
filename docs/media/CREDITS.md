# Media credits

The photographs in `production-detect-*.jpg` come from Wikimedia Commons and are public domain or CC BY (attribution below); nothing is CC BY-SA and nothing comes from the project's own training data. The licences were read from the Commons file pages on 2026-10-04. The other pictures in this folder are charts drawn by [`../figures/make_figures.py`](../figures/make_figures.py) from the numbers of the reports; they contain no third-party content.

| File here | Commons file | Author | Licence |
|---|---|---|---|
| `production-detect-planted-tank-tetras.jpg` | [File:Amaterske akvarium.jpg](https://commons.wikimedia.org/wiki/File:Amaterske_akvarium.jpg) (a small amateur aquarium, 100 litres) | User Aleš Tošovský | public domain |
| `production-detect-cardinal-tetra-school.jpg` | [File:Paracheirodon axelrodi school.jpg](https://commons.wikimedia.org/wiki/File:Paracheirodon_axelrodi_school.jpg) | Tkinias (English Wikipedia) | public domain |
| `production-detect-guppies-planted-tank.jpg` | [File:Маленький заросший аквариум.jpg](https://commons.wikimedia.org/wiki/File:%D0%9C%D0%B0%D0%BB%D0%B5%D0%BD%D1%8C%D0%BA%D0%B8%D0%B9_%D0%B7%D0%B0%D1%80%D0%BE%D1%81%D1%88%D0%B8%D0%B9_%D0%B0%D0%BA%D0%B2%D0%B0%D1%80%D0%B8%D1%83%D0%BC.jpg) | MarDe | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0). Changes: resized and re-compressed, boxes drawn |

Each photo is shown with the boxes that the detector trained in this repository (the exported `best.onnx`) returned when run by the inference service ([`Defish-inference`](https://github.com/GKatzer/Defish-inference)) on 2026-10-04; the number next to a box is the detector score. The pictures were made there (`docs/examples/annotate.py`) and copied here; they carry the licence of the photo they are made from. Photographs resized to at most 1200 px wide.
