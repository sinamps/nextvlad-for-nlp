# nextvlad-for-nlp
This repository contains the code for our paper titled "An Investigation into the Contribution of Locally Aggregated Descriptors to Figurative Language Identification".

For reproducibility and more deterministic execution, run with:
<br>CUBLAS_WORKSPACE_CONFIG=:16:8 python tbinv_earlystop.py


upload the requirements list (pip freeze file)

The current version of the implementation uses a linear scheduler with number of warmups = 2000 and initial learning rate = 1e^-6. However, we also tried other learning rate strategies including cyclic LR and observed no improvements. For the cyclic version, we use base_lr = 1e^-7 and max_lr = 1e^-3.
