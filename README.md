# 说明
## 文件结构：
```python
config.py  #路径、超参数
data_loader.py   #加载数据
main.py   #实体识别训练
ner_evaluate.py   #实体识别训练模型的评估，生成ner_eval_report.txt评估报告
ner_model.py   #实体识别模型结构
ner.predict.py   #实体识别训练模型的预测，配合pending.txt使用
process_cnl2003.py   #对conll2003数据集处理，生成实体识别所需数据
process_TACRED.py   #对Re-TACRED数据集处理，生成实体识别所需数据
process_TACRED_re.py   #对Re-TACRED数据集处理，生成关系抽取所需数据
re_model.py   #关系抽取模型
re_evaluate.py   #关系抽取训练模型的评估，生成re_eval_report.txt评估报告
```
## 数据集：

### CoNLL-2003 数据集
  
CoNLL-2003 是一个经典的命名实体识别（NER）基准数据集，主要用于英语和德语的新闻文本实体标注。该数据集包含四种主要的实体类型：

PER（Person）：人名，如 Barack Obama

ORG（Organization）：组织名，如 United Nations

LOC（Location）：地点名，如 New York

MISC（Miscellaneous）：其他杂项实体，如 Nobel Prize（奖项）、Microsoft Windows（产品）

数据集分为训练集、验证集和测试集，常用于评估 NER 模型的性能。该数据集被广泛用于序列标注模型（如 BiLSTM-CRF、BERT）的基准测试。


### Re-TACRED 数据集
  
Re-TACRED 是 TACRED（TAC Relation Extraction Dataset）的改进版本，专注于关系抽取（RE）任务，旨在更准确地识别文本中实体之间的语义关系。该数据集包含 41 种关系类型（如 "per:employee_of"、"org:founded_by"）和 12 种实体类型：

DATE

DURATION

LOCATION

MISC

MONEY

NUMBER

ORDINAL

ORGANIZATION

PERCENT

PERSON

SET

TIME

相较于原始 TACRED，Re-TACRED 修正了大量标注错误，减少了噪声和歧义，使其成为关系抽取领域更可靠的基准。该数据集适用于训练和评估基于 BERT、SpanBERT 等预训练模型的 RE 方法，广泛应用于信息抽取研究。

## 测试
对两个数据集的ner任务都做了8个epoch的训练，对Re-TACRED的re任务做了5个epoch的训练，评估报告如下：

### conll2003
```
              precision    recall  f1-score   support

       B-LOC     0.8546    0.9059    0.8795      1668
      B-MISC     0.6997    0.7635    0.7302       702
       B-ORG     0.8950    0.7646    0.8247      1661
       B-PER     0.9031    0.8992    0.9011      1617
       I-LOC     0.8139    0.8677    0.8399       257
      I-MISC     0.5676    0.6806    0.6189       216
       I-ORG     0.9039    0.8228    0.8614       835
       I-PER     0.9578    0.9429    0.9503      1156
           O     0.9863    0.9891    0.9877     41776

    accuracy                         0.9669     49888
   macro avg     0.8424    0.8485    0.8438     49888
weighted avg     0.9674    0.9669    0.9669     49888
```

### Re-TACRED_named entity recognition
```
                precision    recall  f1-score   support

        B-DATE     0.9718    0.9615    0.9666      8313
    B-DURATION     0.8844    0.8059    0.8433      2221
    B-LOCATION     0.8829    0.8966    0.8897      9875
        B-MISC     0.7857    0.7538    0.7694      3648
       B-MONEY     0.9159    0.8715    0.8931      1074
      B-NUMBER     0.9080    0.8516    0.8789      5645
     B-ORDINAL     0.9872    0.9585    0.9726       723
B-ORGANIZATION     0.7564    0.7964    0.7759     12171
     B-PERCENT     0.9885    0.9981    0.9933       517
      B-PERSON     0.9036    0.9084    0.9060     18774
         B-SET     0.9860    0.9381    0.9615       226
        B-TIME     0.7670    0.8144    0.7900       388
        I-DATE     0.9632    0.9445    0.9538      4741
    I-DURATION     0.9339    0.9648    0.9491      1875
    I-LOCATION     0.8450    0.8734    0.8590      2472
        I-MISC     0.5619    0.3455    0.4279       880
       I-MONEY     0.9338    0.8575    0.8940      1859
      I-NUMBER     0.6318    0.7801    0.6982       341
     I-ORDINAL     0.0000    0.0000    0.0000         1
I-ORGANIZATION     0.8764    0.8935    0.8849     15444
     I-PERCENT     0.9924    0.9924    0.9924       524
      I-PERSON     0.9462    0.9373    0.9417     12468
         I-SET     0.9138    0.9138    0.9138        58
        I-TIME     0.7891    0.7372    0.7623       274
             O     0.9831    0.9834    0.9833    373353

      accuracy                         0.9619    477865
     macro avg     0.8443    0.8311    0.8360    477865
  weighted avg     0.9619    0.9619    0.9618    477865
```

### Re-TACRED_relation extraction
```
                                       precision    recall  f1-score   support

                        no_relation     0.8494    0.9246    0.8854      7770
                org:alternate_names     0.8582    0.7003    0.7712       337
                 org:city_of_branch     0.7099    0.7209    0.7154       129
              org:country_of_branch     0.6928    0.6928    0.6928       166
                      org:dissolved     1.0000    0.2000    0.3333         5
                        org:founded     0.8293    1.0000    0.9067        34
                     org:founded_by     0.8767    0.7619    0.8153        84
                      org:member_of     0.5000    0.2969    0.3725        64
                        org:members     0.5957    0.4444    0.5091        63
    org:number_of_employees/members     1.0000    0.6154    0.7619        13
org:political/religious_affiliation     0.6522    0.5172    0.5769        29
                   org:shareholders     0.2857    0.1667    0.2105        12
      org:stateorprovince_of_branch     0.6200    0.5439    0.5794        57
          org:top_members/employees     0.8556    0.8034    0.8287       295
                        org:website     0.6667    0.9333    0.7778        30
                            per:age     0.9293    0.8221    0.8724       208
                 per:cause_of_death     1.0000    0.1600    0.2759        50
                        per:charges     0.8165    0.7063    0.7574       126
                       per:children     0.8571    0.6545    0.7423        55
            per:cities_of_residence     0.6750    0.4320    0.5268       125
                  per:city_of_birth     0.8889    0.5333    0.6667        15
                  per:city_of_death     1.0000    0.3846    0.5556        26
         per:countries_of_residence     0.5714    0.4054    0.4743       148
               per:country_of_birth     0.0000    0.0000    0.0000         0
               per:country_of_death     0.0000    0.0000    0.0000        14
                  per:date_of_birth     0.6667    0.8571    0.7500         7
                  per:date_of_death     0.8000    0.5714    0.6667        63
                    per:employee_of     0.6511    0.5452    0.5934       332
                       per:identity     0.9366    0.8703    0.9022      2036
                         per:origin     0.6277    0.5130    0.5646       115
                   per:other_family     0.8462    0.8462    0.8462        52
                        per:parents     0.8679    0.8679    0.8679       106
                       per:religion     0.6667    0.5085    0.5769        59
               per:schools_attended     0.8333    0.6061    0.7018        33
                       per:siblings     0.8571    0.9091    0.8824        66
                         per:spouse     0.8033    0.6712    0.7313        73
       per:stateorprovince_of_birth     0.7000    0.7778    0.7368         9
       per:stateorprovince_of_death     1.0000    0.1875    0.3158        16
  per:stateorprovinces_of_residence     0.6275    0.4384    0.5161        73
                          per:title     0.8554    0.8031    0.8284       523

                           accuracy                         0.8453     13418
                          macro avg     0.7367    0.5848    0.6272     13418
                       weighted avg     0.8422    0.8453    0.8391     13418
```
  
