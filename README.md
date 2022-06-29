## WWW'22 HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization


Authors: Menglin Yang, Min Zhou, Jiahong Liu, Defu Lian, Irwin King

Note: this repository is built upon [HGCF](https://github.com/layer6ai-labs/HGCF) and [HGCN](https://github.com/HazyResearch/hgcn).


## Environment:
The code was developed and tested on the following python environment:
```
python 3.8.15
pytorch 1.5.1
scikit-learn 0.23.2
numpy 1.20.2
scipy 1.6.2
tqdm 4.60.0
```

## Instructions:

Train and evaluation HRCF:

- To evaluate HRCF on Amazon_CD 
  - `bash ./example/run_cd.sh`
- To evaluate HRCF on Amazon_Book
   - `bash ./example/run_book.sh`
- To evaluate HRCF on Yelp
    - `bash ./example/run_yelp.sh`


