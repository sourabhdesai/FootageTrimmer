MAIN_EXE=run
EXE=cli.o footage_trimmer.o
COMPILER=g++
COMPILER_FLAGS= -g -O0 -Wall
LINKER=g++
LDFLAGS = -I/usr/local/include/opencv -lm -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc

all: $(EXE)
	$(LINKER) $(COMPILER_FLAGS) $(EXE) -o $(MAIN_EXE)

cli.o: main.cpp
	$(COMPILER) $(LDFLAGS) $(COMPILER_FLAGS) main.cpp -o cli.o

footage_trimmer.o: FootageTrimmer.h FootageTrimmer.cpp
	$(COMPILER) $(LDFLAGS) $(COMPILER_FLAGS) FootageTrimmer.cpp -o footage_trimmer.o

clean:
	rm -f *.o $(EXE) $(MAIN_EXE)