C
C  f16_deqg.F - Gateway function for f16_deq.f
C
C
C  Usage:  xd=f16_deq(u,x,c);
C
C  To compile and link: fmex f16_deq.f f16_deqg.F
C
C
C  This is the FORTRAN code required for interfacing a .MEX file to MATLAB.
C
C  This subroutine is the main gateway to MATLAB.  When a MEX function
C  is executed MATLAB calls the USRFCN subroutine in the corresponding
C  MEX file.  
C
C  The code down to the solid line remains unchanged for all
C  FORTRAN MEX functions on the sun4 machines.
C
C fintrf.h	- MATLAB/FORTRAN interface header file. This file
C		  contains the declaration of the pointer type needed
C		  by the MATLAB/FORTRAN interface.
C
C
#if defined(__alpha) || defined(cray)
#define pointer integer*8
#define POINTER INTEGER*8
#else
#define pointer integer
#define POINTER INTEGER
#define mxCopyPtrToPtrArray mxCopyPtrToInteger4
#endif
C
C
      SUBROUTINE MEXFUNCTION(NLHS, PLHS, NRHS, PRHS)
      POINTER PLHS(*), PRHS(*)
      INTEGER NLHS, NRHS
C
      POINTER MXCREATEFULL, MXGETPR
      INTEGER MXGETM, MXGETN
C
C  KEEP THE ABOVE SUBROUTINE, ARGUMENT, AND FUNCTION DECLARATIONS FOR USE
C  IN ALL FORTRAN MEX FILES.
C
C---------------------------------------------------------------------
C
      POINTER XDP,UP,XP,CP
      INTEGER NI,NS,NC
C
C  CHECK FOR PROPER NUMBER OF ARGUMENTS.
C
      IF (NRHS .NE. 3) THEN
        CALL MEXERRMSGTXT('Three input arguments are required')
      ELSEIF (NLHS .GT. 1) THEN
        CALL MEXERRMSGTXT('Only one output arguments is computed')
      ENDIF
C
C  FIND DIMENSIONS OF THE INPUT ARGUMENTS.
C
      NI = MXGETM(PRHS(1))
      NDUM = MXGETN(PRHS(1))
      IF (NDUM.GT.NI) NI=NDUM
      NS = MXGETM(PRHS(2))
      NDUM = MXGETN(PRHS(2))
      IF (NDUM.GT.NS) NS=NDUM
      NC = MXGETM(PRHS(3))
      NDUM = MXGETN(PRHS(3))
      IF (NDUM.GT.NC) NC=NDUM
C
C CREATE MATRICES FOR THE OUTPUT ARGUMENTS.
C
      PLHS(1) = MXCREATEFULL(NS,1,0)
C
C  ASSIGN POINTERS TO THE VARIOUS INPUT/OUTPUT PARAMETERS.
C
      XDP = MXGETPR(PLHS(1))
C
      UP = MXGETPR(PRHS(1))
      XP = MXGETPR(PRHS(2))
      CP = MXGETPR(PRHS(3))
C
C  DO THE ACTUAL COMPUTATIONS IN A SUBROUTINE 
C  USING THE POINTERS TO THE MATLAB ARRAYS.
C
      CALL F16_DEQ(%VAL(XDP),%VAL(UP),%VAL(XP),%VAL(CP),NI,NS,NC)
C
      RETURN
      END
