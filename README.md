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

### Training

### Evaluation

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