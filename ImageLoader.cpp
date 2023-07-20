#include "ImageLoader.h"

#include <cstdio>
#include <stdarg.h>

#include <exception>

#include "jpeglib.h"
#include <png.h>
#include <cstring>
#include <iostream>

//-------------------------------------------------------------------------------------------

static void abort_(const char * s, ...) {
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        exit(-1);
}


Image load_png(fs::path file_path)
{
        std::cout << file_path << std::endl;

        Image image;
        const char *file_name = file_path.c_str();

        char header[8];    // 8 is the maximum size that can be checked
        /* open file and test for it being a png */
        FILE *fp = fopen(file_name, "rb");
        if (!fp) 
        {
                throw fs::filesystem_error("load_png failed to open", file_path, std::error_code());
        }
        size_t elementsRead = fread(header, 1, 8, fp);
        if (elementsRead != 8) {
                std::stringstream ss;
                ss << "load_png, File " << file_name << " only read " << elementsRead << " bytes\n";
                throw std::runtime_error(ss.str());
        }
        if (png_sig_cmp((const unsigned char*)header, 0, 8)) {
                std::stringstream ss;
                ss << "load_png, File " << file_name << " is not recognized as a PNG file\n";
                throw std::runtime_error(ss.str());
        }


        /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
        abort_("[read_png_file] png_create_read_struct failed");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
        abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    image.width = png_get_image_width(png_ptr, info_ptr);
    image.height = png_get_image_height(png_ptr, info_ptr);
//     png_byte color_type = png_get_color_type(png_ptr, info_ptr);
//     png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

//     png_byte channels = png_get_channels(png_ptr, info_ptr);

    // printf("color_type:  %d\n", color_type);
    // printf("bit_depth:  %d\n", bit_depth);
    // printf("channels:  %d\n", channels);

//     int number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);


    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during read_image");

    image.stride = png_get_rowbytes(png_ptr,info_ptr);
    image.data.resize(image.stride * image.height);

        png_bytep * row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * image.height);
        for (int y=0; y<image.height; y++) {
                row_pointers[y] = image.data.data() + y * image.stride;
        }
        png_read_image(png_ptr, row_pointers);
        free(row_pointers);

        fclose(fp);

        return image;
}

//-------------------------------------------------------------------------------------------

struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */

  jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct my_error_mgr * my_error_ptr;

static void my_error_exit (j_common_ptr cinfo)
{
        /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
        my_error_ptr myerr = (my_error_ptr) cinfo->err;

        /* Always display the message. */
        /* We could postpone this until after returning, if we chose. */
        (*cinfo->err->output_message) (cinfo);

        /* Return control to the setjmp point */
        longjmp(myerr->setjmp_buffer, 1);
}

//-------------------------------------------------------------------------------------------

Image load_jpeg (fs::path file_path)
{
        Image image;

        const char *filename = file_path.c_str();


  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  /* We use our private extension JPEG error handler.
   * Note that this struct must live as long as the main JPEG parameter
   * struct, to avoid dangling-pointer problems.
   */
  struct my_error_mgr jerr;
  /* More stuff */
  FILE * infile;		/* source file */
  JSAMPARRAY buffer;		/* Output row buffer */
  int row_stride;		/* physical row width in output buffer */

  /* In this example we want to open the input file before doing anything else,
   * so that the setjmp() error recovery below can assume the file is open.
   * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
   * requires it in order to read binary files.
   */

        if ((infile = fopen(filename, "rb")) == NULL) {
                throw fs::filesystem_error("load_jpeg failed to open XXX", file_path, std::error_code());
        }

  /* Step 1: allocate and initialize JPEG decompression object */

  /* We set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object, close the input file, and return.
     */
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    throw std::runtime_error("load_jpeg, jmp happened");
  }
  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */

  jpeg_stdio_src(&cinfo, infile);

  /* Step 3: read file parameters with jpeg_read_header() */

  (void) jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.txt for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  /* Step 5: Start decompressor */

  (void) jpeg_start_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* We may need to do some setup of our own at this point before reading
   * the data.  After jpeg_start_decompress() we have the correct scaled
   * output image dimensions available, as well as the output colormap
   * if we asked for color quantization.
   * In this example, we need to make an output work buffer of the right size.
   */ 
  /* JSAMPLEs per row in output buffer */
  row_stride = cinfo.output_width * cinfo.output_components;
  /* Make a one-row-high sample array that will go away when done with image */
  buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */

  /* Here we use the library's state variable cinfo.output_scanline as the
   * loop counter, so that we don't have to keep track ourselves.
   */
    image.height = cinfo.output_height;
    image.width = cinfo.output_width;
    image.stride = row_stride;
    image.data.resize(image.stride * image.height);
    while (cinfo.output_scanline < cinfo.output_height) {
        /* jpeg_read_scanlines expects an array of pointers to scanlines.
        * Here the array is only one element long, but you could ask for
         * more than one scanline at a time if that's more convenient.
         */
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
        /* Assume put_scanline_someplace wants a pointer and sample count. */
        memcpy(image.data.data() + image.stride*(cinfo.output_scanline-1), buffer[0], row_stride);
    }

  /* Step 7: Finish decompression */

  (void) jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  /* After finish_decompress, we can close the input file.
   * Here we postpone it until after no more JPEG errors are possible,
   * so as to simplify the setjmp error logic above.  (Actually, I don't
   * think that jpeg_destroy can do an error exit, but why assume anything...)
   */
  fclose(infile);

  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */

  /* And we're done! */
  return image;
}

//-------------------------------------------------------------------------------------------
