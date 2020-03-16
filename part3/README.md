### HW2-3: CUDA particles

Log in on Bridges:

Then do:
```
module load cmake/3.11.4
module load cuda

cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./gpu
```

To upload on becourses: put the write-up in the source directory as cs267cs267-hw2-group-parallelBears_hw2_3.pdf and do
```
cmake -DGROUP_NAME=cs267-hw2-group-parallelBears ..
make package
```
