FROM ubuntu:18.04
RUN apt-get update && apt-get clean
RUN apt-get install -y curl
RUN apt-get install -y python3-minimal python3-pip python3-distutils python3-setuptools && apt-get clean
#RUN pip3 install wheel

RUN pip3 install tensorflow==1.12.0
RUN pip3 install flask

ENV BERT_HOME /usr/local/bert/
RUN mkdir $BERT_HOME

WORKDIR $BERT_HOME

##Optimize this later
ADD . $BERT_HOME/

ENTRYPOINT ["/bin/bash", "/usr/local/bert/startup.sh"]

EXPOSE 5555
