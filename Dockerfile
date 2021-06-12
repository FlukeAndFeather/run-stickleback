# syntax=docker/dockerfile:1

# Start with a base Python image
FROM jupyter/scipy-notebook:python-3.8.8
WORKDIR stickleback/

# Install stickleback, data, and harness
RUN pip3 install git+git://github.com/FlukeAndFeather/stickleback.git
COPY . .

CMD [ "python3", "stickleback_test.py", "breaths.pkl", "32", "4" ]
