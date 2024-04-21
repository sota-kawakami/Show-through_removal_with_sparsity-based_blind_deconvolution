#pragma once

#include "Member.h"
#include <vector>

void MakeDirectory(Member &mem);
bool copy(const char* src, const char* dst);

void SaveEvalValue(Member mem);
void SaveImg(Member mem);