# 50.007 Machine Learning, 2016 Fall
Singapore University of Technology and Design
Design Project
Due 7 Dec 2016, 5pm

Group member: You Hongzhou, Yuan Wolong, Yang Wei

## File Descriptions

<!-- **Notice**: if you use the windows system, remember to change to '/' in path strings to '\\' before compiling; you do not need to do so if you use the Mac OS or Linux system. -->

p2.cpp: solution code of Part 2
p3.cpp: solution code of Part 3
p4.cpp: solution code of Part 4
p5.cpp: solution code of Part 5

## Running Instructions

#### Compile:

g++ p2.cpp -o p2 -std=c++11
g++ p3.cpp -o p3 -std=c++11
g++ p4.cpp -o p4 -std=c++11
g++ p5.cpp -o p5 -std=c++11

#### Run:
1. Put the four data folders (EN, CN, SG, ES) under the same directory with the executable files.
2. Run ./p2 ./p3 ./p4 ./p5 separately and finish the training processing.
3. The results will be saved to corresponding folders.

**Notice**: 
If you use the Mac OS or Linux system, you should make sure the encoding method of files in CN is UTF-8(CR-LF, by default) and the encoding method of files in other folders (EN, SG, ES) is UTF-8(LF, transforming command: perl -pi -e 's/\r\n/\n/g' EN/dev.in)

#### Evaluate:

1. Put tevalResult.py under the four test folders.
2. Run python3 evalResult.py dev.out dev.p2.out and so on.






