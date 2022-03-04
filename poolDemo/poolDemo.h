#ifndef __POOLDEMO_H__
#define __POOLDEMO_H__

void poolInit(const char * configPath);
void poolMouseButton(int button, int state, int x, int y);
void poolKey(unsigned char key);
void poolUpdate();
void poolExtrinsicsTransform();

#endif
