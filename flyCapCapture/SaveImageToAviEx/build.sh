g++  -g  SaveImageToAviEx.cpp -lopencv_core -lopencv_highgui `pkg-config opencv --cflags --libs` -I /usr/include/flycapture/ -L /usr/lib -lflycapture -g3 -O3 -o saveToAvi
