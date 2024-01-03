# Bag of tricks for long-tail visual recognition of animal species in camera-trap images

This is the origin TensorFlow implementation for [Bag of tricks for long-tail visual recognition of animal species in camera-trap images](https://doi.org/10.1016/j.ecoinf.2023.102060)

### The Square-root Sampling Branch Framework (SSB)

![SSB - Square-root Sampling Branch Framework](bags4cameratraps/data/ssb.svg?raw=true)

### Requirements

Prepare an environment with python=3.9, tensorflow=2.5.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Datasets

[Caltech Camera Traps](http://lila.science/datasets/caltech-camera-traps), [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti), [WCS Camera Traps](https://lila.science/datasets/wcscameratraps), and [Wellington Camera Traps](https://lila.science/datasets/wellingtoncameratraps) datasets can be downloaded from [Lila](http://lila.science/).

We used the recommended Lila train/val/test partitions for splits based on locations (site). When the test split was not available, the validation set was used as the test set, and a minival set was held out from the training set. The used partitions are available in the `data` folder.

As not all images were used from these datasets, we provide the lists of used images for each dataset [here](https://drive.google.com/drive/folders/1RWIL2695SqNeqB9ZaR5TottVMk8A75_D?usp=sharing).

We also provide scripts for resizing images and converting the dataset to tfrecords format. See the `dataset_tools` folder.

### Training

#### Classifiers

To train a classifier, use the script `train.py`:
```bash
python train.py --model_name=efficientnetv2-b2 \
    --input_size=260 \
    --input_scale_mode=tf \
    --batch_size=64 \
    --epochs=30 \
    --optimizer=adamw \
    --lr=1e-5 \
    --weight_decay=1e-7 \
    --use_cosine_decay \
    --warmup_epochs=2.0 \
    --loss_fn=sce \
    --ra_num_layers=2 \
    --ra_magnitude=9 \
    --train_json=PATH_TO_BE_CONFIGURED/caltech_images_20211210ufam_train.json \
    --val_json=PATH_TO_BE_CONFIGURED/caltech_images_20211210ufam_minival.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED/ \
    --random_seed=42
```

The parameters can also be passed using a config file:
```bash
python train.py --flagfile=configs/efficientnetv2-b2_caltech_representation.config \
    --model_dir=PATH_TO_BE_CONFIGURED
```

For more parameter information, please refer to `train.py`. See the `configs` folder for some training config examples.

#### Two-Stage Training

The `train.py` script is also used to train two-stage methods:
1. Train the first stage by passing the appropriate options to the `train.py` script (e.g., `configs/efficientnetv2-b2_caltech_representation.config`).
2. Use the script `save_base_model.py` to extract the backbone weights (base model) without the classifier.
3. Train the second stage using `train.py` with the appropriate options and pass the backbone weights using the option `--base_model_weights` (e.g., `configs/efficientnetv2-b2_caltech_crt.config`).

#### SSB

SSB is trained in a two-stage fashion:
1. Use the `train.py` script to train a model with the option `--sampling_strategy` set to `instance` and `--freeze_base_model` set to `False`.
2. Save the backbone weights using the `save_base_model.py` script.
3. Train the second stage using the backbone weights from the first stage frozen and the Square-Root resampling strategy: `--sampling_strategy=sqrt --freeze_base_model=True --base_model_weights=PATH_TO_BE_CONFIGURED`.

### Evaluation

To evaluate a classifier, use the script `eval.py`:
```bash
python eval.py --model_name=efficientnetv2-b2 \
    --input_size=260 \
    --num_classes=19 \
    --model_weights=PATH_TO_BE_CONFIGURED/effv2b2_caltech_bboxrepre_10fev/ckp \
    --input_scale_mode=tf \
    --train_json=PATH_TO_BE_CONFIGURED/caltech_images_20211210ufam_train.json \
    --test_json=PATH_TO_BE_CONFIGURED/caltech_images_20211210ufam_val.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED/ \
    --results_file=PATH_TO_BE_CONFIGURED/effv2b2_caltech_bboxrepre_10fev_val_results.json
```

To evaluate SSB, use the option `--model_weights` to pass the weights for the first stage and `--sqrt_model_weights` for the second stage weights (using square-root re-sampling).

### Results

All model checkpoints are available [here](https://drive.google.com/drive/folders/16N9f0Lbdv1p1oXdKOsCn6LKTYtiYqSiP?usp=sharing).

#### Macro F1-Score for two-stage methods using the full image
![Macro F1-Score for two-stage methods using the full image](bags4cameratraps/data/macro_f1_score.png?raw=true)

### Citation

If you find this code useful in your research, please consider citing:

    @article{cunha2023bag,
        title={Bag of tricks for long-tail visual recognition of animal species in camera-trap images},
        author={Cunha, Fagner and dos Santos, Eulanda M and Colonna, Juan G},
        journal={Ecological Informatics},
        volume={76},
        pages={102060},
        year={2023},
        publisher={Elsevier}
    }


### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)