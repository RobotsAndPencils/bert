FROM ubuntu:18.04
RUN apt-get update && apt-get clean && apt-get install -y curl unzip python3-minimal python3-pip
RUN apt-get install -y curl unzip

RUN pip3 install tensorflow==1.12.0
RUN pip3 install flask

ENV BERT_HOME /usr/local/bert/
RUN mkdir $BERT_HOME

WORKDIR $BERT_HOME

RUN curl -O https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
RUN unzip uncased_L-12_H-768_A-12.zip
RUN rm uncased_L-12_H-768_A-12.zip

#RUN curl -O https://s3-us-west-2.amazonaws.com/elasticallen/squad1_base.tar.gz
RUN curl -O https://s3-us-west-2.amazonaws.com/elasticallen/squad2_base.tar.gz
RUN tar xf squad2_base.tar.gz
RUN rm squad2_base.tar.gz

##Optimize this later
ADD . $BERT_HOME/

ENTRYPOINT ["/bin/bash", "/usr/local/bert/startup.sh"]

EXPOSE 5555
