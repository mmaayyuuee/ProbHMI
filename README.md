# # ProbHMI

### **Datasets:**

* Human3.6M:	Download the data from [Human3.6M Dataset](http://vision.imar.ro/human3.6m/description.php), placing it in `./data/h36m_dataset`, and then run the following script:

  ```
  python ./datasets/Human36M/data_format_convert.py 
  ```
* AMASS:  Download the data from [AMASS](https://amass.is.tue.mpg.de/), placing it in `./data/AMASS_Original`, and then run the following script:

  ```
  python ./datasets/AMASS/amass_parser.py
  ```

### **Training:**

* Run `python ./trains/train_gf.py` first, then run  `python ./trains/train_pred.py`

### **Evaluation:**

* Run `python ./evals/eval_for_conformal.py`
