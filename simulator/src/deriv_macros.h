#ifndef __DERIV_MACROS_H__
#define __DERIV_MACROS_H__

    #if prec_deriv == 1

	#define derivPMLdx(matriz, offset, x, z) ((matriz[z][x+1-offset] - matriz[x-offset][z])) 
        #define derivPMLdz(matriz, offset, x, z) ((matriz[z+1-offset][x] - matriz[x][z-offset])) 

    #elif prec_deriv == 2

        #define derivPMLdx(matriz, offset, x, z)\
	    (\
		(-1*matriz[z][x+2-offset]\
		+27*matriz[z][x+1-offset]\
		-27*matriz[z][x-offset]\
		 +1*matriz[z][x-offset-1])\
		/24.0f\
	    ) 
        #define derivPMLdz(matriz, offset, x, z)\
	    (\
		(-1*matriz[z+2-offset][x]\
		+27*matriz[z+1-offset][x]\
		-27*matriz[z-offset][x]\
		 +1*matriz[z-offset-1][x])\
		/24.0f\
	    )

        #define deriv_x(matriz, x, z)\
	    (\
		(-1.0f*matriz[z][x+2]\
		+16.0f*matriz[z][x+1]\
		-30.0f*matriz[z][x+0]\
		+16.0f*matriz[z][x-1]\
		 -1.0f*matriz[z][x-2])\
		/12.0f\
	    ) 
        #define deriv_z(matriz, x, z)\
	    (\
		(-1.0f*matriz[z+2][x]\
		+16.0f*matriz[z+1][x]\
		-30.0f*matriz[z+0][x]\
		+16.0f*matriz[z-1][x]\
		 -1.0f*matriz[z-2][x])\
		/12.0f\
	    ) 

    #elif prec_deriv == 3

        #define derivPMLdx(matriz, offset, x, z)\
	    (\
	 	   (9.0f*matriz[z][x+3-offset]\
	         -125.0f*matriz[z][x+2-offset]\
		+2250.0f*matriz[z][x+1-offset]\
		-2250.0f*matriz[z][x-offset]\
		 +125.0f*matriz[z][x-offset-1]\
		   -9.0f*matriz[z][x-offset-2])\
		   /1920.0f\
	    ) 
        #define derivPMLdz(matriz, offset, x, z)\
	    (\
		   (9.0f*matriz[z+3-offset][x]\
 	         -125.0f*matriz[z+2-offset][x]\
	        +2250.0f*matriz[z+1-offset][x]\
	        -2250.0f*matriz[z-offset][x]\
	         +125.0f*matriz[z-offset-1][x]\
	           -9.0f*matriz[z-offset-2][x])\
		   /1920.0f\
	    ) 

    #elif prec_deriv == 4
	#define derivPMLdx(matriz, offset, x, z)\
	(\
		-6.975446e-04*matriz[z][x+4-offset]\
		+9.570313e-03*matriz[z][x+3-offset]\
		-7.975260e-02*matriz[z][x+2-offset]\
		+1.196289e+00*matriz[z][x+1-offset]\
		-1.196289e+00*matriz[z][x+0-offset]\
		+7.975260e-02*matriz[z][x-1-offset]\
		-9.570313e-03*matriz[z][x-2-offset]\
		+6.975446e-04*matriz[z][x-3-offset]\
	)


	#define derivPMLdz(matriz, offset, x, z)\
	(\
		-6.975446e-04*matriz[z+4-offset][x]\
		+9.570313e-03*matriz[z+3-offset][x]\
		-7.975260e-02*matriz[z+2-offset][x]\
		+1.196289e+00*matriz[z+1-offset][x]\
		-1.196289e+00*matriz[z+0-offset][x]\
		+7.975260e-02*matriz[z-1-offset][x]\
		-9.570313e-03*matriz[z-2-offset][x]\
		+6.975446e-04*matriz[z-3-offset][x]\
	)

        #define deriv_x(matriz, x, z)\
	    (\
		(-9.0f*matriz[z][x+4]\
		+128.0f*matriz[z][x+3]\
		-1008.0f*matriz[z][x+2]\
		+8064.0f*matriz[z][x+1]\
		-14350.0f*matriz[z][x+0]\
		+8064.0f*matriz[z][x-1]\
		-1008.0f*matriz[z][x-2]\
		+128.0f*matriz[z][x-3]\
		-9.0f*matriz[z][x-4])\
		/5040.0f\
	    ) 
        #define deriv_z(matriz, x, z)\
	    (\
		(-9.0f*matriz[z+4][x]\
		+128.0f*matriz[z+3][x]\
		-1008.0f*matriz[z+2][x]\
		+8064.0f*matriz[z+1][x]\
		-14350.0f*matriz[z+0][x]\
		+8064.0f*matriz[z-1][x]\
		-1008.0f*matriz[z-2][x]\
		+128.0f*matriz[z-3][x]\
		-9.0f*matriz[z-4][x])\
		/5040.0f\
	    ) 


    #elif prec_deriv == 5

	#define derivPMLdx(matriz, offset, x, z)\
	(\
		+1.186795e-04*matriz[z][x+5-offset]\
		-1.765660e-03*matriz[z][x+4-offset]\
		+1.384277e-02*matriz[z][x+3-offset]\
		-8.972168e-02*matriz[z][x+2-offset]\
		+1.211243e+00*matriz[z][x+1-offset]\
		-1.211243e+00*matriz[z][x+0-offset]\
		+8.972168e-02*matriz[z][x-1-offset]\
		-1.384277e-02*matriz[z][x-2-offset]\
		+1.765660e-03*matriz[z][x-3-offset]\
		-1.186795e-04*matriz[z][x-4-offset]\
	)


	#define derivPMLdz(matriz, offset, x, z)\
	(\
		+1.186795e-04*matriz[z+5-offset][x]\
		-1.765660e-03*matriz[z+4-offset][x]\
		+1.384277e-02*matriz[z+3-offset][x]\
		-8.972168e-02*matriz[z+2-offset][x]\
		+1.211243e+00*matriz[z+1-offset][x]\
		-1.211243e+00*matriz[z+0-offset][x]\
		+8.972168e-02*matriz[z-1-offset][x]\
		-1.384277e-02*matriz[z-2-offset][x]\
		+1.765660e-03*matriz[z-3-offset][x]\
		-1.186795e-04*matriz[z-4-offset][x]\
	)

    #elif prec_deriv == 6

	#define derivPMLdx(matriz, offset, x, z)\
	(\
		-2.184781e-05*matriz[z][x+6-offset]\
		+3.590054e-04*matriz[z][x+5-offset]\
		-2.967290e-03*matriz[z][x+4-offset]\
		+1.744766e-02*matriz[z][x+3-offset]\
		-9.693146e-02*matriz[z][x+2-offset]\
		+1.221336e+00*matriz[z][x+1-offset]\
		-1.221336e+00*matriz[z][x+0-offset]\
		+9.693146e-02*matriz[z][x-1-offset]\
		-1.744766e-02*matriz[z][x-2-offset]\
		+2.967290e-03*matriz[z][x-3-offset]\
		-3.590054e-04*matriz[z][x-4-offset]\
		+2.184781e-05*matriz[z][x-5-offset]\
	)


	#define derivPMLdz(matriz, offset, x, z)\
	(\
		-2.184781e-05*matriz[z+6-offset][x]\
		+3.590054e-04*matriz[z+5-offset][x]\
		-2.967290e-03*matriz[z+4-offset][x]\
		+1.744766e-02*matriz[z+3-offset][x]\
		-9.693146e-02*matriz[z+2-offset][x]\
		+1.221336e+00*matriz[z+1-offset][x]\
		-1.221336e+00*matriz[z+0-offset][x]\
		+9.693146e-02*matriz[z-1-offset][x]\
		-1.744766e-02*matriz[z-2-offset][x]\
		+2.967290e-03*matriz[z-3-offset][x]\
		-3.590054e-04*matriz[z-4-offset][x]\
		+2.184781e-05*matriz[z-5-offset][x]\
	)


    #elif prec_deriv == 8

	#define derivPMLdx(matriz, offset, x, z)\
	(\
		-8.52346420e-07*matriz[z][x+8-offset]\
		+1.70217111e-05*matriz[z][x+7-offset]\
		-1.66418878e-04*matriz[z][x+6-offset]\
		+1.07727117e-03*matriz[z][x+5-offset]\
		-5.34238560e-03*matriz[z][x+4-offset]\
		+2.30363667e-02*matriz[z][x+3-offset]\
		-1.06649846e-01*matriz[z][x+2-offset]\
		+1.23409107e+00*matriz[z][x+1-offset]\
		-1.23409107e+00*matriz[z][x+0-offset]\
		+1.06649846e-01*matriz[z][x-1-offset]\
		-2.30363667e-02*matriz[z][x-2-offset]\
		+5.34238560e-03*matriz[z][x-3-offset]\
		-1.07727117e-03*matriz[z][x-4-offset]\
		+1.66418878e-04*matriz[z][x-5-offset]\
		-1.70217111e-05*matriz[z][x-6-offset]\
		+8.52346420e-07*matriz[z][x+7-offset]\
	)
	
	#define derivPMLdz(matriz, offset, x, z)\
	(\
		-8.52346420e-07*matriz[z+8-offset][x]\
		+1.70217111e-05*matriz[z+7-offset][x]\
		-1.66418878e-04*matriz[z+6-offset][x]\
		+1.07727117e-03*matriz[z+5-offset][x]\
		-5.34238560e-03*matriz[z+4-offset][x]\
		+2.30363667e-02*matriz[z+3-offset][x]\
		-1.06649846e-01*matriz[z+2-offset][x]\
		+1.23409107e+00*matriz[z+1-offset][x]\
		-1.23409107e+00*matriz[z+0-offset][x]\
		+1.06649846e-01*matriz[z-1-offset][x]\
		-2.30363667e-02*matriz[z-2-offset][x]\
		+5.34238560e-03*matriz[z-3-offset][x]\
		-1.07727117e-03*matriz[z-4-offset][x]\
		+1.66418878e-04*matriz[z-5-offset][x]\
		-1.70217111e-05*matriz[z-6-offset][x]\
		+8.52346420e-07*matriz[z+7-offset][x]\
	)
/*
        #define deriv_x(matriz, x, z)\
	(\
		-2.42812743e-06*matriz[z][x+8]\
		+5.07429079e-05*matriz[z][x+7]\
		-5.18000518e-04*matriz[z][x+6]\
		+3.48096348e-03*matriz[z][x+5]\
		-1.76767677e-02*matriz[z][x+4]\
		+7.54208754e-02*matriz[z][x+3]\
		-3.11111111e-01*matriz[z][x+2]\
		+1.77777778e+00*matriz[z][x+1]\
		-3.05484410e+00*matriz[z][x+0]\
		+1.77777778e+00*matriz[z][x-1]\
		-3.11111111e-01*matriz[z][x-2]\
		+7.54208754e-02*matriz[z][x-3]\
		-1.76767677e-02*matriz[z][x-4]\
		+3.48096348e-03*matriz[z][x-5]\
		-5.18000518e-04*matriz[z][x-6]\
		+5.07429079e-05*matriz[z][x-7]\
		-2.42812743e-06*matriz[z][x-8]\
	) 

        #define deriv_z(matriz, x, z)\
	(\
		-2.42812743e-06*matriz[z+8][x]\
		+5.07429079e-05*matriz[z+7][x]\
		-5.18000518e-04*matriz[z+6][x]\
		+3.48096348e-03*matriz[z+5][x]\
		-1.76767677e-02*matriz[z+4][x]\
		+7.54208754e-02*matriz[z+3][x]\
		-3.11111111e-01*matriz[z+2][x]\
		+1.77777778e+00*matriz[z+1][x]\
		-3.05484410e+00*matriz[z+0][x]\
		+1.77777778e+00*matriz[z-1][x]\
		-3.11111111e-01*matriz[z-2][x]\
		+7.54208754e-02*matriz[z-3][x]\
		-1.76767677e-02*matriz[z-4][x]\
		+3.48096348e-03*matriz[z-5][x]\
		-5.18000518e-04*matriz[z-6][x]\
		+5.07429079e-05*matriz[z-7][x]\
		-2.42812743e-06*matriz[z-8][x]\
	) 
*/

        #define deriv_x(matriz, x, z)\
	    (\
		(-9.0f*matriz[z][x+4]\
		+128.0f*matriz[z][x+3]\
		-1008.0f*matriz[z][x+2]\
		+8064.0f*matriz[z][x+1]\
		-14350.0f*matriz[z][x+0]\
		+8064.0f*matriz[z][x-1]\
		-1008.0f*matriz[z][x-2]\
		+128.0f*matriz[z][x-3]\
		-9.0f*matriz[z][x-4])\
		/5040.0f\
	    ) 
        #define deriv_z(matriz, x, z)\
	    (\
		(-9.0f*matriz[z+4][x]\
		+128.0f*matriz[z+3][x]\
		-1008.0f*matriz[z+2][x]\
		+8064.0f*matriz[z+1][x]\
		-14350.0f*matriz[z+0][x]\
		+8064.0f*matriz[z-1][x]\
		-1008.0f*matriz[z-2][x]\
		+128.0f*matriz[z-3][x]\
		-9.0f*matriz[z-4][x])\
		/5040.0f\
	    ) 


    #endif
#endif
