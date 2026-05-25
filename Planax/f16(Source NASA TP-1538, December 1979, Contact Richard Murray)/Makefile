# Makefile for F16 simulation (matlab mex format)
# RMM, 1 Nov 95
#

#
# Create the mex file for the F16 model
#
f16_deq.mex: f16_deq.f f16_deqg.F
	fmex f16_deq.f f16_deqg.F

#
# Clean up the directory of extraneous data
#
clean:
	/bin/rm -f *.o *.mex

#
# Rule to create a tar file of the relevant files
#
FILES = f16_deq.f f16_deqg.F f16_deq.m Makefile
f16.tar: $(FILES)
	tar cf $@ $(FILES)
		