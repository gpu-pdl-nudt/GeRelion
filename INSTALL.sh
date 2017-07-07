#!/usr/bin/env sh

#Some flags variables
BUILD_FFTW=true
BUILD_FLTK=true
BUILD_RELION=true
N_THREADS=$@

#Use single-precision instead of the default of double-precision?
FLOAT_PRECISION=true

# Set the param below to "false" if you do not have an MPI installation and only want to build the sequential version of RELION
HAVE_MPI=true
# Set the param below to "false" if you have trouble compiling fltk and still want to build RELION without the GUI
HAVE_FLTK=true

#Some path variables
# Note that as of RELION-1.3, the prefix actually needs to be RELION_HOME. You can move the bin and lib directories elsewhere after building
GERELION_HOME=$PWD
PREFIX=$GERELION_HOME

#External libraries versions
VFFTW=fftw-3.2.2
VFLTK=fltk-1.3.0

# Some other vars
GREEN="\033[32m"
ENDC="\033[0m"



#################### FFTW ###########################
# In case Fortran compilation fails consider adding  --disable-fortran to the ./configure line below. 
if $BUILD_FFTW; then
  echo -e "$GREEN Compiling $VFFTW ...$ENDC"
  echo -e "See $GERELION_HOME/external/fftw_build.log for details"
  cd external
  tar -zxf $VFFTW.tar.gz
  cd $VFFTW
  if $FLOAT_PRECISION; then
    float_option=" --enable-float "
  else
    float_option=""
  fi
  ./configure $float_option --enable-threads --enable-shared prefix=$PREFIX > $GERELION_HOME/external/fftw_build.log
  make $N_THREADS >> $GERELION_HOME/external/fftw_build.log 
  make install >> $GERELION_HOME/external/fftw_build.log 
  cd ../..
fi

#################### FLTK ###########################
if $BUILD_FLTK; then
  echo -e "$GREEN Compiling $VFLTK ...$ENDC"
  echo -e "See $GERELION_HOME/external/fltk_build.log for details"
  cd external
  tar -zxf $VFLTK.tar.gz
  cd $VFLTK
  ./configure --enable-shared prefix=$PREFIX > $GERELION_HOME/external/fltk_build.log
  make $N_THREADS >> $GERELION_HOME/external/fltk_build.log
  make install >> $GERELION_HOME/external/fltk_build.log
  cd ../..
fi

#################### RELION ###########################
if $BUILD_GERELION; then
  echo -e "$GREEN Compiling gerelion ...$ENDC"
  echo -e "See $GERELION_HOME/gerelion_build.log for details"
 if $HAVE_FLTK; then
  fltk_cxx=`$PREFIX/bin/fltk-config --cxxflags`
  fltk_ld=`$PREFIX/bin/fltk-config --ldflags`
 else
  fltk_cxx=""
  fltk_ld=""
 fi
 #if $HAVE_MPI; then
 # ./configure prefix=$PREFIX --enable-mpi CPPFLAGS="-I$PREFIX/include $fltk_cxx"  LDFLAGS="-L$PREFIX/lib $fltk_ld" > $RELION_HOME/relion_build.log
 #else
 # ./configure prefix=$PREFIX CPPFLAGS="-I$PREFIX/include $fltk_cxx"  LDFLAGS="-L$PREFIX/lib $fltk_ld" > $RELION_HOME/relion_build.log
 #fi
 make $N_THREADS >> $GERELION_HOME/gerelion_build.log
 #make install >> $RELION_HOME/relion_build.log
 #mv $RELION_HOME/bin/relion_maingui $PREFIX/bin/relion 
 #cp $RELION_HOME/scripts/qsub.csh $PREFIX/bin/qsub.csh
fi

echo "Done!"

