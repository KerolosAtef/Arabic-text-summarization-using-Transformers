# Overview

This project mainly focuses on Arabic text summarization using transformers.
After that we use the summary result with its original text to be
evaluated through arabic classification and clustering problems 
to check whether the meaning of the summary matches the original text.

The classification problem is article classification between 5 topics context.</br>
The clustering problem is the same arabic articles clustering between 5 topics.</br>


# Dependencies

- nltk
- Scipy
- pickle
- transformers==4.19.2
- tensorflow-gpu==2.9.1
- numpy
- pandas
- re
- time
- PyQt5
- pyarabic
- farasapy
- functools
- operator
- emoji
- string
- sklearn
- plotly

# Datasets

- [Arabic News Articles](https://www.kaggle.com/datasets/haithemhermessi/sanad-dataset) :
  - For Classification and clustering.
  - 45,000 Articles with 7 different topics.
- [WikiLingua](https://github.com/esdurmus/Wikilingua) :
  - For Summarization
  - ~ 40,000 Arabic articles with their summaries.

# Dealing with project files

The project folder contains the following files:

- [summarization.ipynb](#summarization)
- [inference.py](#inference)
- [class_clust.ipynb](#class-clust)
- [class_clust_infer.py](#class-clust-infer)
- [MainWindow.py](#main-window)
- [Arabic_stop_words.txt](#arabic-stop-words)
- [champion_models.pickle](#champion-models)
- [objects.pickle](#objects)

 Also it contains the following folders:

- [arabic Folder](#arabic-folder)
- [checkpoints Folder](#checkpoints-folder)

## Description

---

- ### **summarization**

    This file contains all the processes of summarization algorithm

- ### **class-clust**

    This file contains all the processes of building clustering and classification models

- ### **inference**

    This file contains the inference code for summarization that returns the summary of the text to be summarized in the GUI.

- ### **class-clust-infer**

    This file contains the inference code for classification and clustring to be imported in the GUI code file that returns the class and cluster names of the original and summarized text in addition to their similarity score.

- ### **main-window**

    This is the GUI code file that is used for inference.

- ### **arabic-stop-words**

    This is a text file that is used in the preprocessing process in summarization file.

- ### **champion-models**

    This pickle file contains the TF-IDF vectorizer, champion classfier and champion cluster to be loaded in class_clust_infer.py. It has a large size, so you can [click here](https://drive.google.com/uc?export=view&id=1uNF0miG1U0STR2WBRPUQY2AGDqIsOEes) for download.

- ### **objects**

    This pickle file contains the trained tokenizers to be loaded in inference.py file. It has a large size, so you can [click here](https://drive.google.com/uc?export=view&id=11iKqwXhQmKIeE0O967TpnAYAJrRILFVQ) for download.

- ### **arabic-folder**

    This file contains the WikiLingua arabic datasets. You can [click here](https://drive.google.com/uc?export=view&id=1tAgP9xR0iPuOYxYgGbPC0tm7lQW-UOtj) for download.

    If you downloaded it, you wouldn't need to run **read_text(dir_path, fin)** in summarization Notebook.

- ### **checkpoints-folder**

    This file is generated from summarization code file for saving the checkpoints while training in addition to be loaded in inference.py to use the latest checkpoints directly for inference. You can [click here](https://drive.google.com/uc?export=view&id=11kAuaZ_N3OpKM-ep6c3dup-kQ97cWm-e) for download.

    If you downloaded it, you wouldn't need to run **Training steps** section in summarization notebook.

# GUI Output
![image](https://drive.google.com/uc?export=view&id=1fg2912fPqHLKFrC5Z75Ldco8V3f_P7y4)
