# - try to find FreeImage library and include files
#  FREEIMAGE_INCLUDE_DIR, where to find FreeImage.h, etc.
#  FREEIMAGE_LIBRARIES, the libraries to link against
#  FREEIMAGE_FOUND, If false, do not try to use FreeImage.


IF (WIN32)
  #message(STATUS "GLUT include dir: ${GLUT_INCLUDE_DIR}")
  FIND_PATH( FREEIMAGE_INCLUDE_DIR NAMES FreeImage.h PATHS ${FREEIMAGE_DIR} ${GLUT_INCLUDE_DIR} )
  FIND_LIBRARY( FREEIMAGE_FreeImage_LIBRARY NAMES FreeImage PATHS ${FREEIMAGE_DIR} )
  IF (NOT FREEIMAGE_INCLUDE_DIR)
    SET(FREEIMAGE_DIR "FreeImage" CACHE PATH "Path to FreeImage header and library files")
  ENDIF (NOT FREEIMAGE_INCLUDE_DIR)
ELSE (WIN32)
  
  IF (APPLE)
    # These values for Apple could probably do with improvement.
    FIND_PATH( FREEIMAGE_INCLUDE_DIR FreeImage.h
      /System/Library/Frameworks/FreeImage.framework/Versions/A/Headers
      /usr/include
      /usr/local/include
      )
    SET(FREEIMAGE_FreeImage_LIBRARY "-lfreeimage" CACHE STRING "FreeImage library for OSX") 
  ELSE (APPLE)
    
    FIND_PATH( FREEIMAGE_INCLUDE_DIR FreeImage.h
      /usr/include
      /usr/openwin/share/include
      /usr/openwin/include
      /opt/graphics/OpenGL/include
      /opt/graphics/OpenGL/contrib/libglut
      )
  
    FIND_LIBRARY( FREEIMAGE_FreeImage_LIBRARY NAMES FreeImage freeimage
      /usr/lib
      /usr/openwin/lib
      )
     
  ENDIF (APPLE)
  
ENDIF (WIN32)

SET( FREEIMAGE_FOUND "NO" )
IF(FREEIMAGE_INCLUDE_DIR)
  IF(FREEIMAGE_FreeImage_LIBRARY)
    SET( FREEIMAGE_LIBRARIES ${FREEIMAGE_FreeImage_LIBRARY} )
    SET( FREEIMAGE_FOUND "YES" )
    
    #The following deprecated settings are for backwards compatibility with CMake1.4
    SET (FREEIMAGE_LIBRARY ${FREEIMAGE_LIBRARIES})
    SET (FREEIMAGE_INCLUDE_PATH ${FREEIMAGE_INCLUDE_DIR})
    
  ENDIF(FREEIMAGE_FreeImage_LIBRARY)
ENDIF(FREEIMAGE_INCLUDE_DIR)

MARK_AS_ADVANCED(
  FREEIMAGE_INCLUDE_DIR
  FREEIMAGE_LIBRARIES
  )
