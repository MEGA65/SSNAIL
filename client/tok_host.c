/* Host test for ssnail_tokc.c: tok_host image.ssnail < lines
   prints each line's token ids, comma-separated, then its folded text */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ssnail_tokc.h"

static unsigned char *img;
static long img_len;
static uint32_t load_base;
#define BASE 0x8000000UL

void tm_read(uint32_t a, uint8_t *dst, uint8_t n)
{
  uint8_t i;
  for (i = 0; i < n; i++, a++) {
    long o = a < BASE + 256 ? (long)(a - BASE) : (long)(256 + a - load_base);
    dst[i] = (o >= 0 && o < img_len) ? img[o] : 0;
  }
}

int main(int argc, char **argv)
{
  FILE *f = fopen(argv[1], "rb");
  char line[400];
  static uint32_t ids[400];
  if (!f) return 1;
  fseek(f, 0, SEEK_END); img_len = ftell(f); fseek(f, 0, SEEK_SET);
  img = malloc(img_len); fread(img, 1, img_len, f); fclose(f);
  load_base = img[0x64] | img[0x65] << 8 | img[0x66] << 16 | (uint32_t)img[0x67] << 24;
  if (!load_base) load_base = BASE + 256;
  tok_init(BASE);
  while (fgets(line, sizeof line, stdin)) {
    uint16_t n, i;
    char out[2048], *o = out;
    out[0] = 0;
    line[strcspn(line, "\n")] = 0;
    n = tok_encode(line, 1, ids, 400);
    for (i = 0; i < n; i++) printf(i ? ",%u" : "%u", (unsigned)ids[i]);
    printf("\n");
    fold_reset();
    for (i = 0; i < n; i++) {
      uint8_t b[255], l = tok_piece(ids[i], b);
      fold_feed(b, l, o);
      o += strlen(o);
    }
    printf("%s\n", out);
  }
  return 0;
}
