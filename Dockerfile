# syntax=docker/dockerfile:1

# Start with a base Python image
FROM jupyter/scipy-notebook:python-3.8.8
WORKDIR stickleback/

# Install stickleback, data, and harness
RUN pip3 install git+git://github.com/FlukeAndFeather/stickleback.git
COPY lunges_subset.pkl lunges_subset.pkl
COPY stickleback_test.py stickleback_test.py

CMD [ "python3", "stickleback_test.py", "lunges_subset.pkl", "32", "4" ]
