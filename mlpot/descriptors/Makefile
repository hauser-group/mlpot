SHELL = /bin/bash

CC= g++
CPPFLAGS= -fPIC -g -O3 -Wall -ffast-math -lm -fopenmp -std=c++11
LDFLAGS= -shared -fopenmp

SOURCES = $(shell echo *.cpp)
HEADERS = $(shell echo *.h)
OBJECTS=$(SOURCES:.cpp=.o)

TARGET=libDescriptorSet.so

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
