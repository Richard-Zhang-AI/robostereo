set -x
set -e

# sudo apt install libgl1 libopengl0 libgl1-mesa-dri libgl1-mesa-glx libosmesa6-dev ffmpeg libosmesa6

cd dependencies
cd opensora
pip install -v -e .
cd ..

cd openvla-oft
bash ./install_mujoco.sh
pip install --no-deps -e .
pip install --no-deps git+https://github.com/moojink/dlimp_openvla
pip install --no-deps "git+https://github.com/moojink/transformers-openvla-oft.git"
cd ..


# git clone https://github.com/volcengine/verl.git
# cd verl
# pip install --no-deps -e .
# cd ..

if [ ! -d robosuite ]; then
	git clone https://github.com/ARISE-Initiative/robosuite.git
fi
cd robosuite
git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
pip install --no-deps -e .
cd ..

if [ ! -d robomimic ]; then
	git clone https://github.com/ARISE-Initiative/robomimic.git
fi
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install --no-deps -e .
cd ..

if [ ! -d mimicgen ]; then
	git clone https://github.com/NVlabs/mimicgen.git
fi
cd mimicgen
pip install --no-deps -e .
cd ..

if [ ! -d robosuite-task-zoo ]; then
	git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
fi
cd robosuite-task-zoo
git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
pip install --no-deps -e .
cd ..


echo "installed all dependencies"
