reset
g++ arTrackPointGrey.cpp -I /usr/include/flycapture/ -L /usr/lib -lflycapture -lGL -lglut -lGLU -o arTrack `pkg-config opencv cvblob --cflags --libs`
