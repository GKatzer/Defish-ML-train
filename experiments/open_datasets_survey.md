# Open datasets for Defish: a survey (September 2026)

Method: a search of the web and of Hugging Face; the pages were opened through WebFetch.
What did not open (Kaggle gave only a title, Roboflow gave HTTP 403) is marked **"not checked"**: the figures are taken
from search snippets, and the licence has to be read by a person before use.

## Main conclusion

1. **I did not find open datasets of diseases of specifically aquarium fish with reliable labels.** Everything I found is about
   ponds and aquaculture (India, Bangladesh, Korea). Your 189 crops, checked by a fish keeper, remain the reference (gold).
2. **Most of the "different" disease datasets are one and the same set**, multiplied across platforms
   (7 classes × ~250–300 pictures). The real unique volume is of the order of 2 thousand pictures, and it has to be deduplicated by
   hash before use, otherwise the metrics will be inflated just like they were by your `dup_` copies.
3. For the **detector** there is much more open data than for diseases (Open Images, DeepFish, FishNet and others), but almost
   all of it is underwater footage or fishing, not home aquariums.

## A. Classification of diseases

| Dataset | What is inside | Licence | Match with your classes | Verdict |
|---|---|---|---|---|
| [Freshwater Fish Disease Aquaculture in South Asia](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia) (Kaggle, subirbiswas19) | 7 classes: Bacterial Red disease, Aeromoniasis, Bacterial gill disease, Saprolegniasis, Parasitic diseases, White tail disease (a virus), Healthy. The description says 250 per class; collected at an agricultural university and on a farm in Odisha (India) "with the help of experts" | **not checked** | Saprolegniasis ≈ dermatomycosis; Parasitic ≈ partly oodiniosis; Healthy = healthy. The rest has no counterpart | The main candidate for silver. Download by hand from Kaggle (a login is needed) |
| [panda992/fish_disease_datasets](https://huggingface.co/datasets/panda992/fish_disease_datasets) (HF) | The same 7 classes, **2 450 pictures** (train 2 082 / test 368); the README only lists the classes. **Downloaded and checked**, see the section "Check of panda992" below | **not stated** | as above | A copy of the Kaggle set. **A weak silver:** small pictures, built-in augmentations, duplicates, a foreign domain |
| Rkaaaa/, dhevadharsan-d/, TuVH/`fish_disease_datasets` (HF) | Repositories of the same name and the same size | not stated | as above | Most likely copies of the same set (the content was not compared) |
| [Saon110/bd-fish-disease-dataset](https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset) (HF, gated) | 5 887 pictures: 7 fish classes with 290–303 each + 4 shrimp classes; assembled from Mendeley (shrimp) and an HF collection of fish diseases | **CC BY-NC-SA 4.0**: non-commercial use only + share-alike | as above | The fish part is, again, the same set. **NC licence: do not take it if the application may become commercial** |
| Roboflow Universe: [Fish Diseases (ornamental-fish)](https://universe.roboflow.com/ornamental-fish/fish-diseases) | 914 pictures of "ornamental fish", classes Bacterial / Fungal / Healthy / Parasitic / White Tail | **not checked** (403) | as above | The class names coincide with the South Asian set; probably derived from it, and not a real aquarium (this is my hypothesis, not checked). Your observation that the labelling is worse on Roboflow agrees with this |
| Roboflow Universe: fishlens-modelv1, Van Anh (EUS/Fin_rot/MAS/swim_bladder), Fish disease 721 and others | Classes: Columnaris, Bacterial Red, EUS, Bacterial gill, Fungal, Ich (white spots), Streptococcus, TiLV, Fin_rot, swim_bladder… | **not checked** | Ich ↔ oodiniosis, Fin_rot ↔ fin rot, Columnaris ↔ fin rot/dermatomycosis | Ponds and tilapia, the quality of the labelling is unknown. Look only after the main set |
| FlatIMG ([arXiv 2407.11348](https://arxiv.org/pdf/2407.11348)) | Sick flatfish from 10 Korean farms, labelled | not checked | none | Marine aquaculture, not suitable |
| Zenodo [15479903](https://zenodo.org/records/15479903), [15434271](https://zenodo.org/records/15434271) | **These are conference papers (PDF), there is no data** | CC BY 4.0 | — | Exclude |
| HF ybli/yolo-fish-object-and-fish-disease-recognition | The description is in Chinese, the data is on an external link | not stated | unknown | Skip |

A useful pointer for further search: [alzayats/fish-datasets](https://github.com/alzayats/fish-datasets) (a list of fish datasets on GitHub).

### Check of panda992 (downloaded to `$DEFISH_DATA_DIR/external/panda992_fish_disease_datasets`, unpacked to `images/`, table `manifest.csv`)

- **Resolution:** 1 610 pictures of 128×128 and 821 pictures of 224×224, the rest are a handful of non-standard sizes; the median JPEG is 3.7 KB.
  Your crops are an order of magnitude larger; small signs (spots, coating, a torn fin) are not visible at 128 px.
- **Duplicates:** 1 747 of the 2 450 are unique by hash, that is exactly ~250 per class, as in the Kaggle description. The rest (703 pictures) are
  exact copies. **163 pictures are in train and test at the same time**: a leak inside the set itself. There are also ≥101 pairs of near-duplicates
  (dHash); this is a lower bound: rotated and cropped variants are not caught by this method.
- **Augmentations are baked into the data:** the sample shows stretched edges, rotations, overexposure. "250 per class" is most likely not 250 different fish.
- **Label conflict:** the same picture occurs under two different classes (1 hash).
- **Domain:** ponds, markets, fish in hands, opened gills, some of them dead. There are practically no aquarium fish.
- **"Shortcut" risk:** the "healthy" ones are whole fish on grass or on a white background, the "sick" ones are close-ups of wounds and gills. A classifier
  may learn "close-up → disease" instead of the disease itself.
- **Outcome:** usable at most as an auxiliary source of embeddings and for checking "a healthy fish on a foreign background → there must be no disease".
  The labels cannot be taken as a reference. Use it only by the rule: if it does not improve the metric on your reference, throw it away.

## B. Fish detection and "non-aquarium" fish (for the detector and for checking false alarms)

| Dataset | What is inside | Licence | What for |
|---|---|---|---|
| Open Images (class Fish, from memory: check that it is there) | Flickr photos: a fish on a plate, at a market, in hands, in water, on a white background | annotations CC BY 4.0, images CC BY 2.0 | A variety of scenes: it cures the problem "not an aquarium → disease". The main candidate |
| [DeepFish](https://www.nature.com/articles/s41597-022-01416-0) | ~40 thousand underwater frames, 20 habitats of Australia, ~15 thousand boxes on 4 505 pictures, masks exist | CC BY 4.0 | The look of fish in different conditions; the domain is underwater, not an aquarium |
| [Aquarium Dataset](https://public.roboflow.com/object-detection/aquarium) (Roboflow public) | 638 pictures from two public aquariums in the USA, class `fish` and others | not checked | Many small fish behind glass, closer to your domain |
| Fish4Knowledge, OzFish | Underwater frames of Taiwan (27 230 pictures, 23 species) and Australia (~43 thousand boxes) | not checked | Small fish, low resolution |
| [FishNet (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.html) | 94 532 pictures, 17 357 species, boxes exist | not checked | A variety of species; a large set |
| iNaturalist | Photos of species by amateurs | depends on the photo (some are CC BY-NC) | Only with a licence filter |

## C. What I suggest next

1. Download the public `panda992/fish_disease_datasets`, **deduplicate it by hash** and check on your reference
   whether it gives a gain as silver (the rule: if the metric on the reference does not grow, the set is thrown away).
2. Build the detector's "pool of non-aquarium fish" from Open Images + DeepFish + Aquarium Dataset, checking the licences.
3. Do not use Saon110 (NC) before a decision about the commercial fate of the project.
