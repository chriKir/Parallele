// auto_levels.c
// adjusts input image levels so that it is spread over the full 0-255 range
// works separately on each image component (e.g. RGB)
// Peter Thoman

#include "stb_image.h"
#include "stb_image_write.h"
#include "time_ms.h"

#define MAX_COMPONENTS 4

int main(int argc, char** argv) {
	if(argc != 3) {
		printf("Usage: auto_levels [inputfile] [outputfile]\nExample: auto_levels test.png test_adjusted.png\n");
		return -1;
	}

	int width, height, components;
	unsigned char *data = stbi_load(argv[1], &width, &height, &components, 0);

	unsigned long long start_time = time_ms();

	// determine maximum, minimum and average ---------------------------------
	// initialize
	unsigned char min_val[MAX_COMPONENTS], max_val[MAX_COMPONENTS], avg_val[MAX_COMPONENTS];
	unsigned long long sum[MAX_COMPONENTS];
	for(int c = 0; c<components; ++c) {
		min_val[c] = 255; // start min with largest possible value
		max_val[c] = 0;   // start max with smallest possible value
		sum[c] = 0;
	}
	// iterate
	for(int x=0; x<width; ++x) {
		for(int y=0; y<height; ++y) {
			for(int c=0; c<components; ++c) {
				unsigned char val = data[c + x*components + y*width*components];
				if(val<min_val[c]) min_val[c] = val;
				if(val>max_val[c]) max_val[c] = val;
				sum[c] += val;
			}
		}
	}
	// calculate average and multiplicative factors
	float min_fac[MAX_COMPONENTS], max_fac[MAX_COMPONENTS];
	for(int c = 0; c<components; ++c) {
		avg_val[c] = (unsigned char)(sum[c]/((unsigned long long)width*height));
		min_fac[c] = (float)avg_val[c] / ((float)avg_val[c] - (float)min_val[c]);
		max_fac[c] = (255.0f-(float)avg_val[c]) / ((float)max_val[c] - (float)avg_val[c]);
		printf("Component %1u: %3u/%3u/%3u * %3.2f/%3.2f\n", c, min_val[c],avg_val[c],max_val[c], min_fac[c],max_fac[c]);
	}

	// adjust image -----------------------------------------------------------
	// iterate
	for(int x=0; x<width; ++x) {
		for(int y=0; y<height; ++y) {
			for(int c=0; c<components; ++c) {
				int index = c + x*components + y*width*components;
				unsigned char val = data[index];
				float v = (float)val - (float)avg_val[c];
				v *= (val < avg_val[c]) ? min_fac[c] : max_fac[c];
				data[index] = (unsigned char)(v + avg_val[c]);
			}
		}
	}

	printf("Done, took %16lu ms\n", time_ms()-start_time);

	stbi_write_png(argv[2], width, height, components, data, width*components);
	stbi_image_free(data);
}
