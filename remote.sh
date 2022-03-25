source ~/code/robotics/carla-collect/py37trajectron/bin/activate

# CPLEX is optional
export CPLEX_STUDIO_DIR1210=/opt/ibm/ILOG/CPLEX_Studio1210

jupyter notebook \
	--NotebookApp.iopub_data_rate_limit=1.0e10 \
	--no-browser \
	--ip 192.168.1.131 \

