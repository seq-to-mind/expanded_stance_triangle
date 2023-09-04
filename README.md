## Introduction
* One implementation of the paper __Guiding Computational Stance Detection with Expanded Stance Triangle Framework__ in ACL2023 <br>
* This repo and the enriched annotation are only for research use. Please cite the papers if they are helpful. <br>

## Data Format
+ The original SemEval16 Tweet Stance Detection (Task A) data is located at `data/SemEval16_Tweet_Stance_Detection/Original_Annotation`.<br>
+ The enriched annotation of the SemEval16 Tweet Stance Detection (Task A) is located at `data/SemEval16_Tweet_Stance_Detection/Enriched_Annotation`.<br>
+ Each row in the data file is one sample. For instance, here is one row:<br>
`1 <segmenter>  Climate Change is a Real Concern <s> We blame cities for the majority of CO2 emissions without acknowledging their vulnerability to #CFCC15 #journey2015 #S2228`<br>
+ `<segmenter>` is the delimiter between the stance label and model input.<br>
+ The stance label value 1 denotes '__Favor__', 0 denotes '__Against__', and 2 denotes '__None__'.<br>

## Citation
If the work is helpful, please cite our papers in your reports, slides, and papers.<br>

```
@inproceedings{liu-etal-2023-guiding,
    title = "Guiding Computational Stance Detection with Expanded Stance Triangle Framework",
    author = "Liu, Zhengyuan  and
      Yap, Yong Keong  and
      Chieu, Hai Leong  and
      Chen, Nancy",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.220",
    doi = "10.18653/v1/2023.acl-long.220",
    pages = "3987--4001",
}

```

