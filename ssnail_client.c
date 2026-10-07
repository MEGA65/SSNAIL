/*
  SSNAIL MEGA65 Client 0.1
*/

#include "includes.h"
#include <string.h>
#include <stdio.h>
#include <mega65/dirent.h>
#include <mega65/fileio.h>

#include <mega65/targets.h>

#define SSNAIL_FASTIO_BASE 0xFFD7500L
#define HEADER_DESCRIPTION_OFFSET 0x80

struct ModelInfo {
    char filename[32];
    char path[64];
    uint32_t mem_needed;
    char description[64];
};

struct ModelInfo models[20];
uint8_t num_models = 0;

uint32_t available_ram = 0;

void configure_ssnail_ram(void) {
    uint8_t target = detect_target();
    uint8_t hr_mb = 8;
    uint8_t sd_base = 0;
    uint8_t sd_size = 0;
    
    // R4 and later have SDRAM that we can map
    if (target >= TARGET_MEGA65R4 && target != TARGET_SIMULATION && target != TARGET_EMULATION) {
        sd_base = 0x88; // $8800000 base
        sd_size = 64;   // 64 MB SDRAM
        available_ram = (hr_mb + sd_size) * 1024L * 1024L;
    } else {
        available_ram = hr_mb * 1024L * 1024L;
    }
    
    lpoke(SSNAIL_FASTIO_BASE + 0x10, hr_mb);
    lpoke(SSNAIL_FASTIO_BASE + 0x11, sd_base);
    lpoke(SSNAIL_FASTIO_BASE + 0x12, sd_size);
}

void screen_stash(void)
{
  lcopy(0xf000,0x40000L,80*50);
  lcopy(0xff80000L,0x40000L+80*50,80*50);
}

void screen_restore(void)
{
  lcopy(0x40000L,0xf000,80*50);
  lcopy(0x40000L+80*50,0xff80000L,80*50);
}

void print_box(unsigned char x1, unsigned char y1,
               unsigned char x2, unsigned char y2,
               unsigned char colour)
{
  uint16_t char_addr = 0xf000 + y1 * 80;
  for(int x=x1;x<=x2;x++) {
    POKE(char_addr+x,0x20);
    lpoke(0xff80000L - 0xf000 + char_addr+x, 0x20 | colour);    
  }
  for(int y=y1+1;y<y2;y++) {
    char_addr+=80;
    POKE(char_addr+x1,0x20);
    lpoke(0xff80000L - 0xf000 + char_addr+x1, 0x20 | colour);    
    POKE(char_addr+x2,0x20);
    lpoke(0xff80000L - 0xf000 + char_addr+x2, 0x20 | colour);        
  }
  char_addr+=80;
  for(int x=x1;x<=x2;x++) {
    POKE(char_addr+x,0x20);
    lpoke(0xff80000L - 0xf000 + char_addr+x, 0x20 | colour);    
  }
}

void print_text80(unsigned char x, unsigned char y, unsigned char colour, char *msg)
{
  uint16_t char_addr = 0xf000 + x + y * 80;
  while (*msg) {
    uint8_t char_code = *msg;
    POKE(char_addr + 0, char_code);
    lpoke(0xff80000L - 0xf000 + char_addr, colour);
    msg++;
    char_addr += 1;
  }
}

void graphics_clear_screen(void)
{
  lfill(0x40000L, 0, 32768L);
  lfill(0x48000L, 0, 32768L);
}

void h640_text_mode(void)
{
  // lower case
  POKE(0xD018, 0x16);

  // Normal text mode
  POKE(0xD054, 0x00);
  // H640, V400, fast CPU, extended attributes
  POKE(0xD031, 0xE8);
  // Adjust D016 smooth scrolling for VIC-III H640 offset
  POKE(0xD016, 0xC9);
  // 80 chars per line logical screen layout
  POKE(0xD058, 80);
  POKE(0xD059, 80 / 256);
  // Draw 80 chars per row
  POKE(0xD05E, 80);
  // Put 4KB screen at $F000
  POKE(0xD060, 0x00);
  POKE(0xD061, 0xf0);
  POKE(0xD062, 0x00);

  // 50 lines of text
  POKE(0xD07B, 50);

  lfill(0xf000, 0x20, 4000);
  // Clear colour RAM, while setting all chars to 4-bits per pixel
  lfill(0xff80000L, 0x0E, 4000);
}

void assess_ssnail(void) {
    uint8_t id = lpeek(SSNAIL_FASTIO_BASE + 0x00);
    uint8_t version = lpeek(SSNAIL_FASTIO_BASE + 0x01);
    uint8_t caps = lpeek(SSNAIL_FASTIO_BASE + 0x02);
    
    char msg[80];
    if (id == 0x53) {
        sprintf(msg, "SSNAIL found: v%d, caps: 0x%02X", version, caps);
        print_text80(0, 2, 0x0A, msg);
        
        uint8_t hr_size = lpeek(SSNAIL_FASTIO_BASE + 0x10);
        uint8_t sd_base = lpeek(SSNAIL_FASTIO_BASE + 0x11);
        uint8_t sd_size = lpeek(SSNAIL_FASTIO_BASE + 0x12);
        sprintf(msg, "Memory: HyperRAM %d MB, SDRAM Base %d MB, SDRAM Size %d MB", 
                hr_size, sd_base, sd_size);
        print_text80(0, 3, 0x0A, msg);
        
        available_ram = ((uint32_t)hr_size + (uint32_t)sd_size) * 1024L * 1024L;
    } else {
        sprintf(msg, "SSNAIL not found! ID is 0x%02X (expected 0x53)", id);
        print_text80(0, 2, 0x02, msg);
        available_ram = 0;
    }
}

#include "snail_sprites.h"

void setup_snail_sprite(void) {
    // Copy Frame 0 to $E000 (left) and $E0C0 (right) initially
    for(int i=0; i<168; i++) {
        POKE(0xE000 + i, snail_frame0_left[i]);
        POKE(0xE0C0 + i, snail_frame0_right[i]);
    }
    
    // Set pointers for Sprites 1 and 2
    POKE(0xF3F9, 0xE000 / 64); // Sprite 1 (Left)
    POKE(0xF3FA, 0xE0C0 / 64); // Sprite 2 (Right)
    
    // Positions (center screen, 32px wide total)
    POKE(0xD002, 160);      // Sprite 1 X
    POKE(0xD003, 140);      // Sprite 1 Y
    POKE(0xD004, 160 + 16); // Sprite 2 X
    POKE(0xD005, 140);      // Sprite 2 Y
    
    // Clear MSB of X for sprites 1 and 2
    POKE(0xD010, PEEK(0xD010) & 0xF9);
    
    // Disable double size
    POKE(0xD01D, 0); 
    POKE(0xD017, 0); 
    
    // Enable 64px width and 16-color mode for sprites 1 and 2 (bits 1 and 2 = 0x06)
    POKE(0xD057, PEEK(0xD057) | 0x06); // SPRX64EN
    POKE(0xD074, PEEK(0xD074) | 0x06); // SPR16EN
    
    // Load Palette (Colors 1-14)
    // Sprite 1 uses palette slice 1 ($Dx10)
    // Sprite 2 uses palette slice 2 ($Dx20)
    // Color 15 is cycled via primary color registers ($D028, $D029)
    for (int i = 1; i < 15; i++) {
        // Sprite 1
        POKE(0xD110 + i, snail_palette_r[i]);
        POKE(0xD210 + i, snail_palette_g[i]);
        POKE(0xD310 + i, snail_palette_b[i]);
        
        // Sprite 2
        POKE(0xD120 + i, snail_palette_r[i]);
        POKE(0xD220 + i, snail_palette_g[i]);
        POKE(0xD320 + i, snail_palette_b[i]);
    }
    
    // Enable sprites 1 and 2
    POKE(0xD015, PEEK(0xD015) | 0x06);
    
    // Initial Color for the 'F' pixels
    POKE(0xD028, 0x0E); // Sprite 1 primary color
    POKE(0xD029, 0x0E); // Sprite 2 primary color
}

void hide_snail_sprite(void) {
    POKE(0xD015, PEEK(0xD015) & 0xF9);
}

void load_model_data(char *path, char *filename) {
    char fullpath[128];
    sprintf(fullpath, "%s%s%s", path, (strcmp(path, "/") == 0) ? "" : "/", filename);
    
    char msg[80];
    sprintf(msg, "Loading %s...", filename);
    print_text80(0, 46, 0x0E, msg);
    
    uint8_t fd = open(fullpath);
    if (fd == 0xff) {
        print_text80(0, 46, 0x02, "Failed to open model file.");
        return;
    }
    
    uint8_t buf[512];
    size_t n = read512(buf);
    if (n < 256) { close(fd); return; }
    
    uint32_t host_base = buf[0x70] | (buf[0x71]<<8) | ((uint32_t)buf[0x72]<<16) | ((uint32_t)buf[0x73]<<24);
    uint32_t host_len  = buf[0x74] | (buf[0x75]<<8) | ((uint32_t)buf[0x76]<<16) | ((uint32_t)buf[0x77]<<24);
    uint32_t load_base = buf[0x64] | (buf[0x65]<<8) | ((uint32_t)buf[0x66]<<16) | ((uint32_t)buf[0x67]<<24);
    uint32_t payload_len = buf[0x2C] | (buf[0x2D]<<8) | ((uint32_t)buf[0x2E]<<16) | ((uint32_t)buf[0x2F]<<24);
    
    uint32_t offsets[3] = { 0x8000000, host_base, load_base };
    uint32_t lengths[3] = { 256, host_len, payload_len };
    uint32_t total_length = 256 + host_len + payload_len;
    
    uint8_t chunk_idx = 0;
    uint32_t chunk_written = 0;
    uint32_t total_written = 0;
    
    int buf_pos = 0;
    int buf_avail = n;
    
    uint8_t pulse_colours[] = {6, 14, 3, 1, 3, 14};
    uint8_t pulse_idx = 0;
    
    setup_snail_sprite();
    
    uint32_t last_anim_time = PEEK(0xA2) * 65536L + PEEK(0xA1) * 256L + PEEK(0xA0);
    uint8_t anim_frame = 0;
    
    while (chunk_idx < 3) {
        if (chunk_written == lengths[chunk_idx]) {
            chunk_idx++;
            chunk_written = 0;
            if (chunk_idx >= 3) break;
            if (lengths[chunk_idx] == 0) continue;
            
            // set new pointer
            uint32_t dest = offsets[chunk_idx];
            lpoke(SSNAIL_FASTIO_BASE + 0x14, dest & 0xFF);
            lpoke(SSNAIL_FASTIO_BASE + 0x15, (dest >> 8) & 0xFF);
            lpoke(SSNAIL_FASTIO_BASE + 0x16, (dest >> 16) & 0xFF);
            lpoke(SSNAIL_FASTIO_BASE + 0x17, ((dest >> 24) & 0x0F) | 0x40);
        }
        
        if (buf_avail == 0) {
            n = read512(buf);
            if (n == 0) break;
            buf_pos = 0;
            buf_avail = n;
        }
        
        while ((lpeek(SSNAIL_FASTIO_BASE + 0x17) & 0x80) == 0) {
            // Wait for READY
        }
        
        uint32_t to_write = lengths[chunk_idx] - chunk_written;
        if (to_write > buf_avail) to_write = buf_avail;
        
        for (int i=0; i<to_write; i++) {
            lpoke(SSNAIL_FASTIO_BASE + 0x18, buf[buf_pos + i]);
        }
        
        chunk_written += to_write;
        buf_pos += to_write;
        buf_avail -= to_write;
        total_written += to_write;
        
        // 1Hz Animation Check
        uint32_t now = PEEK(0xA2) * 65536L + PEEK(0xA1) * 256L + PEEK(0xA0);
        if (now - last_anim_time >= 50) { // approx 1 second
            last_anim_time = now;
            anim_frame = 1 - anim_frame;
            
            const uint8_t *src_left = anim_frame ? snail_frame1_left : snail_frame0_left;
            const uint8_t *src_right = anim_frame ? snail_frame1_right : snail_frame0_right;
            
            for(int i=0; i<168; i++) {
                POKE(0xE000 + i, src_left[i]);
                POKE(0xE0C0 + i, src_right[i]);
            }
        }
        
        if (total_written % 131072 == 0 || total_written == total_length) {
            float mb_written = (float)total_written / 1048576.0f;
            float mb_total = (float)total_length / 1048576.0f;
            int percent = (int)((float)total_written * 100.0f / (float)total_length);
            
            sprintf(msg, "Loading %s... [%d%%] %.1f / %.1f MB        ", filename, percent, mb_written, mb_total);
            print_text80(0, 46, 0x0E, msg);
            
            POKE(0xD028, pulse_colours[pulse_idx]);
            POKE(0xD029, pulse_colours[pulse_idx]);
            pulse_idx = (pulse_idx + 1) % 6;
        }
    }
    
    // Final flush commit
    lpoke(SSNAIL_FASTIO_BASE + 0x14, 0);
    lpoke(SSNAIL_FASTIO_BASE + 0x15, 0);
    lpoke(SSNAIL_FASTIO_BASE + 0x16, 0);
    lpoke(SSNAIL_FASTIO_BASE + 0x17, 0);
    
    close(fd);
    
    hide_snail_sprite();
    
    sprintf(msg, "Loaded %s successfully.                      ", filename);
    print_text80(0, 46, 0x0A, msg);
}

void inspect_model(char *path, char *filename) {
    if (num_models >= 20) return;

    char fullpath[128];
    sprintf(fullpath, "%s%s%s", path, (strcmp(path, "/") == 0) ? "" : "/", filename);
    
    uint8_t fd = open(fullpath);
    if (fd != 0xff) {
        uint8_t header[512];
        size_t bytes = read512(header);
        close(fd);

        if (bytes >= 256 && header[0] == 'S' && header[1] == 'S' && header[2] == 'N' && header[3] == 'L') {
            struct ModelInfo *m = &models[num_models];
            strncpy(m->filename, filename, sizeof(m->filename)-1);
            m->filename[sizeof(m->filename)-1] = '\0';
            strncpy(m->path, path, sizeof(m->path)-1);
            m->path[sizeof(m->path)-1] = '\0';
            
            // H_MEM_NEEDED is at 0x38 (little-endian)
            m->mem_needed = header[0x38] | (header[0x39] << 8) | ((uint32_t)header[0x3A] << 16) | ((uint32_t)header[0x3B] << 24);
            
            // Read optional description from 0x80
            strncpy(m->description, (char*)&header[HEADER_DESCRIPTION_OFFSET], sizeof(m->description)-1);
            m->description[sizeof(m->description)-1] = '\0';
            
            num_models++;
        }
    }
}

void list_models(char *path) {
    chdirroot();
    if (strcmp(path, "/SSNAIL") == 0 || strcmp(path, "/SSNAIL/") == 0) {
        chdir("SSNAIL");
    }
    
    unsigned char dir = opendir();
    if (dir != 0xff) {
        struct m65_dirent *d;
        while ((d = readdir(dir))) {
            int len = strlen(d->d_name);
            if (len > 7 && strcmp(d->d_name + len - 7, ".ssnail") == 0) {
                inspect_model(path, d->d_name);
            }
        }
        closedir(dir);
    }
}

void display_models(uint8_t start_line) {
    char msg[80];
    for (int i=0; i<num_models; i++) {
        sprintf(msg, "%d. %s (%.1f MB)", i+1, models[i].filename, models[i].mem_needed / (1024.0 * 1024.0));
        print_text80(2, start_line + i*2, 0x0A, msg);
        if (models[i].description[0]) {
            sprintf(msg, "   %s", models[i].description);
            print_text80(2, start_line + i*2 + 1, 0x0B, msg);
        }
    }
}

struct ModelInfo* select_default_model() {
    struct ModelInfo *best_model = NULL;
    uint32_t max_size = 0;
    
    for (int i=0; i<num_models; i++) {
        if (models[i].mem_needed <= available_ram) {
            if (models[i].mem_needed > max_size) {
                max_size = models[i].mem_needed;
                best_model = &models[i];
            }
        }
    }
    return best_model;
}

uint8_t term_x = 0;
uint8_t term_y = 48;

void read_line(char *buf, int max_len) {
    int pos = 0;
    while(1) {
        if (PEEK(0xD610)) {
            uint8_t c = PEEK(0xD610);
            POKE(0xD610, 0);
            
            if (c == 0x0D) { // Return
                buf[pos] = '\0';
                break;
            } else if (c == 0x14) { // Delete
                if (pos > 0) {
                    pos--;
                    term_x--;
                    POKE(0xf000 + term_y * 80 + term_x, ' ');
                }
            } else if (c >= 32 && c < 127 && pos < max_len - 1) {
                buf[pos++] = c;
                POKE(0xf000 + term_y * 80 + term_x, c);
                lpoke(0xff80000L + term_y * 80 + term_x, 0x0E);
                term_x++;
            }
        }
    }
}

void load_config_default_model(char *buf, int max_len) {
    buf[0] = '\0';
    chdirroot();
    chdir("SSNAIL");
    
    uint8_t fd = open("config.txt");
    if (fd != 0xff) {
        uint8_t file_buf[512];
        size_t bytes = read512(file_buf);
        close(fd);
        
        if (bytes > 0) {
            int i = 0;
            while (i < bytes && file_buf[i] != '\n' && file_buf[i] != '\r' && i < max_len - 1) {
                buf[i] = file_buf[i];
                i++;
            }
            buf[i] = '\0';
        }
    }
}

void test_execution(void) {
    char msg[80];
    
    // Read H_CODE, H_TOKENS from header
    uint32_t h_code = lpeek(0x8000008) | (lpeek(0x8000009)<<8) | ((uint32_t)lpeek(0x800000A)<<16) | ((uint32_t)lpeek(0x800000B)<<24);
    uint32_t h_tokens = lpeek(0x8000014) | (lpeek(0x8000015)<<8) | ((uint32_t)lpeek(0x8000016)<<16) | ((uint32_t)lpeek(0x8000017)<<24);
    
    // Configure runtime block
    // R_POS = 0x40
    lpoke(0x8000040, 0); lpoke(0x8000041, 0); lpoke(0x8000042, 0); lpoke(0x8000043, 0);
    // R_PROMPT_LEN = 0x44 (set to 1)
    lpoke(0x8000044, 1); lpoke(0x8000045, 0); lpoke(0x8000046, 0); lpoke(0x8000047, 0);
    // R_N_GENERATE = 0x48 (set to 1)
    lpoke(0x8000048, 1); lpoke(0x8000049, 0); lpoke(0x800004A, 0); lpoke(0x800004B, 0);
    
    // Write a dummy token to H_TOKENS
    lpoke(h_tokens, 1); lpoke(h_tokens+1, 0); lpoke(h_tokens+2, 0); lpoke(h_tokens+3, 0);
    
    // Set Job Pointer ($08-$0B)
    lpoke(SSNAIL_FASTIO_BASE + 0x08, h_code & 0xFF);
    lpoke(SSNAIL_FASTIO_BASE + 0x09, (h_code >> 8) & 0xFF);
    lpoke(SSNAIL_FASTIO_BASE + 0x0A, (h_code >> 16) & 0xFF);
    lpoke(SSNAIL_FASTIO_BASE + 0x0B, (h_code >> 24) & 0xFF);
    
    print_text80(0, term_y - 2, 0x0E, "Running test execution...                   ");
    
    // Start timing
    uint32_t t_start = PEEK(0xA2) * 65536L + PEEK(0xA1) * 256L + PEEK(0xA0);
    
    // GO! (Write 1 to $03)
    lpoke(SSNAIL_FASTIO_BASE + 0x03, 0x01);
    
    // Wait for busy to clear
    while (lpeek(SSNAIL_FASTIO_BASE + 0x04) & 0x01) {
        // Loop
    }
    
    uint32_t t_end = PEEK(0xA2) * 65536L + PEEK(0xA1) * 256L + PEEK(0xA0);
    uint8_t status = lpeek(SSNAIL_FASTIO_BASE + 0x04);
    uint8_t err = lpeek(SSNAIL_FASTIO_BASE + 0x05);
    
    if (err) {
        sprintf(msg, "Execution failed! Error code: %d", err);
        print_text80(0, term_y - 2, 0x02, msg);
    } else {
        uint32_t jiffies = t_end - t_start;
        // ~50 or 60 jiffies per second, let's assume 50 for PAL
        float secs = (float)jiffies / 50.0f;
        if (secs < 0.01) secs = 0.01; // Avoid div by zero
        sprintf(msg, "Success! Token took ~%.2f seconds (%.1f tok/s)", secs, 1.0f / secs);
        print_text80(0, term_y - 2, 0x0A, msg);
    }
}

int main(void)
{
  mega65_io_enable();

  POKE(0xd020,0);
  POKE(0xd021,0);  
  
  h640_text_mode();

  print_text80(0, 0, 0x0F, "SSNAIL Chat Client");

  assess_ssnail();

  print_text80(0, 5, 0x0E, "Scanning SD card for models...");
  list_models("/");
  list_models("/SSNAIL/");
  
  // Clear scanning message
  print_text80(0, 5, 0x0E, "                              ");

  display_models(6);

  struct ModelInfo *current_model = NULL;
  char default_model_name[80];
  load_config_default_model(default_model_name, sizeof(default_model_name));
  
  if (default_model_name[0] != '\0') {
      for (int i=0; i<num_models; i++) {
          if (strcmp(models[i].filename, default_model_name) == 0 && models[i].mem_needed <= available_ram) {
              current_model = &models[i];
              break;
          }
      }
  }
  
  if (!current_model) {
      current_model = select_default_model();
  }
  
  char msg[80];
  if (current_model) {
      sprintf(msg, "Selected default model: %s", current_model->filename);
      print_text80(0, term_y - 2, 0x0A, msg);
      load_model_data(current_model->path, current_model->filename);
  } else {
      print_text80(0, term_y - 2, 0x02, "No suitable model found fitting in RAM.");
  }

  char input_buf[80];
  while(1) {
      print_text80(0, term_y, 0x0F, "> ");
      term_x = 2;
      read_line(input_buf, sizeof(input_buf));
      
      // Clear line
      lfill(0xf000 + term_y * 80, ' ', 80);
      
      if (input_buf[0] == '/') {
          if (strncmp(input_buf, "/model ", 7) == 0) {
              char *name = input_buf + 7;
              int found = 0;
              for (int i=0; i<num_models; i++) {
                  if (strcmp(models[i].filename, name) == 0) {
                      current_model = &models[i];
                      lfill(0xf000 + (term_y - 2) * 80, ' ', 80); // Clear line
                      load_model_data(current_model->path, current_model->filename);
                      found = 1;
                      break;
                  }
              }
              if (!found) {
                  print_text80(0, term_y - 2, 0x02, "Model not found.            ");
              }
          } else if (strncmp(input_buf, "/test", 5) == 0) {
              if (current_model) {
                  test_execution();
              } else {
                  print_text80(0, term_y - 2, 0x02, "No model loaded.            ");
              }
          }
      } else {
          // Normal chat logic
          if (current_model) {
              sprintf(msg, "Sending to %s: %s", current_model->filename, input_buf);
              print_text80(0, term_y - 1, 0x0B, msg);
          }
      }
  }
  
  return 0;
}
