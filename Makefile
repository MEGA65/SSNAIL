# Build SSNAIL images from GGUF models.
#
#   make foo.ssnail                      # from foo.gguf
#   CONTEXT_WINDOW=128 make foo.ssnail
#   make foo.ssnail WTYPE=q4_0 ATTICRAM=8
#   make foo.run PROMPT="Once upon a time" TOKENS=60     # convert + emulate
#   make foo.chat                        # interactive session on the emulator
#   make foo.chat TEMP=0                 # greedy, exactly as the v1 hardware
#   make foo.chat HW=1                   # hardware numerics instead of float
#   make foo.info                        # architecture, tokenizer, tensors
#   make foo.tokcheck                    # image tokenizer vs the GGUF's, on sample text
#   make foo.all                         # foo.8mb.ssnail, foo.64mb.ssnail, foo.72mb.ssnail
#   make foo.72mb.ssnail                 # one size explicitly
#
# Memory sizes:
#    8mb  attic RAM only (all boards)
#   64mb  model in SDRAM at $8800000 + 64 KB attic RAM host window (R4-R6)
#   72mb  model from $8010000 through attic RAM into SDRAM (R4-R6)
# >8mb images are loaded via SSNAIL through a staging buffer; see README.
# foo.ssnail is the same as foo.8mb.ssnail unless ATTICRAM is set.
#   make findmodels [QUERY="tinystories minilm"] [MEM=8]   # search Hugging Face
#   make allmodels                       # fetch the list's missing GGUFs into models/,
#                                        # then images of each in sdcard/SSNAIL/ (copy
#                                        # that folder to the SD card's root)
#   make modellist [QUERY=...]           # (re)write models/models.txt from a search
#   make fetch URL=https://huggingface.co/.../resolve/main/foo.gguf
#   make test
#   make client                          # the MEGA65 chat front end: client/ssnailchat.prg,
#                                        # and sdcard/SSNCHAT.D81 holding it (llvm-mos)
#   make tokcheck-c                      # the client's tokenizer vs ssnail_tok.py, on
#                                        # every image in MODELS_OUT
#
# The converter reads the architecture (llama, gpt2, bert) and tokenizer from
# the GGUF itself, so one rule covers every model.  Images depend on the
# tools, so they rebuild when the converter changes.
#
# Variables (environment or command line):
#   CONTEXT_WINDOW  context length / max sequence (default: min(model, 256), 128 for bert)
#   WTYPE           auto (default: best that fits; asks before going below the file's
#                   own precision), keep, mixed, f32, f16, bf16, q8_0, q4_0
#   FALLBACK        format for non-native weights with WTYPE=keep (default q8_0)
#   ATTICRAM        memory for plain foo.ssnail: 8, 64 or 72 (default 8);
#                   HYPERRAM_MB is still accepted
#   PROMPT, TOKENS  for the .run target
#   TEXT            text file for .tokcheck (default: built-in sample)
#   MODELS_DIR      allmodels' GGUF cache (default models)
#   MODELS_OUT      allmodels' images (default sdcard/SSNAIL): foo.ssnail at the
#                   default context, and foo.ctxN.ssnail for each N in MODELS_CTX
#                   the model was trained for (ATTN takes at most 1024 positions)
#   MODELS_CTX      default 512 1024
#   MODELS_MEM      allmodels' memory size: 8, 64 or 72 (default 64: R4-R6 SDRAM)

SSNAIL_TOOLS ?= $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON       ?= python3
CONVERT      := $(PYTHON) $(SSNAIL_TOOLS)ssnail_convert.py
SIM          := $(PYTHON) $(SSNAIL_TOOLS)ssnail_sim.py
TOOL_DEPS    := $(addprefix $(SSNAIL_TOOLS),ssnail_convert.py ssnail_isa.py)

WTYPE        ?= auto
FALLBACK     ?= q8_0
ATTICRAM     ?= $(or $(HYPERRAM_MB),8)
PROMPT       ?= Once upon a time
TOKENS       ?= 120
# TEMP: leave unset for the chat defaults (sampling); 0 = greedy, as the hardware
CHAT_OPTS     = -n $(TOKENS) $(if $(TEMP),$(if $(filter 0 0.0,$(TEMP)),--greedy,--temp $(TEMP))) $(if $(HW),--hw)

CONVERT_OPTS  = --wtype $(WTYPE) --fallback $(FALLBACK) --mem-mb $(ATTICRAM) $(if $(TYPE),--type "$(TYPE)")
CONVERT_OPTS += $(if $(CONTEXT_WINDOW),--ctx $(CONTEXT_WINDOW))

MODELS_DIR   ?= models
MODELS_OUT   ?= sdcard/SSNAIL
MODELS_CTX   ?= 512 1024
MODELS_MEM   ?= 64
MODEL_LIST   := $(MODELS_DIR)/models.txt
# (a split model's parts are not one GGUF the converter can read)
MODEL_GGUFS   = $(filter-out %-of-00002.gguf %-of-00003.gguf %-of-00004.gguf %-of-00005.gguf,\
                $(wildcard $(MODELS_DIR)/*.gguf))

LLVM_MOS     ?= /opt/llvm-mos
MOS_CC        = $(LLVM_MOS)/bin/mos-mega65-clang
LIBC          = $(SSNAIL_TOOLS)mega65-libc
CLIENT_SRC    = $(addprefix $(SSNAIL_TOOLS)client/,ssnail_chat.c ssnail_tokc.c ssnail_tokc.h)

.PHONY: test clean findmodels fetch modellist allmodels allmodels-images client tokcheck-c %.all %.chat %.run %.tokcheck
.PRECIOUS: %.ssnail

%.ssnail: %.gguf $(TOOL_DEPS)
	$(CONVERT) convert $< -o $@ $(CONVERT_OPTS)

SIZED_OPTS = --wtype $(WTYPE) --fallback $(FALLBACK) $(if $(CONTEXT_WINDOW),--ctx $(CONTEXT_WINDOW)) $(if $(TYPE),--type "$(TYPE)")

%.8mb.ssnail: %.gguf $(TOOL_DEPS)
	$(CONVERT) convert $< -o $@ $(SIZED_OPTS) --mem-mb 8

%.64mb.ssnail: %.gguf $(TOOL_DEPS)
	$(CONVERT) convert $< -o $@ $(SIZED_OPTS) --mem-mb 64

%.72mb.ssnail: %.gguf $(TOOL_DEPS)
	$(CONVERT) convert $< -o $@ $(SIZED_OPTS) --mem-mb 72

# Build every size; sizes the model can't fit are skipped, not fatal.
%.all: %.gguf
	-@$(MAKE) -k --no-print-directory -f $(firstword $(MAKEFILE_LIST)) \
	  $*.8mb.ssnail $*.64mb.ssnail $*.72mb.ssnail

%.run: %.ssnail
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_chat.py $< $(CHAT_OPTS) --prompt "$(PROMPT)" 

%.chat: %.ssnail
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_chat.py $< $(CHAT_OPTS)

%.tokcheck: %.ssnail %.gguf
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_tok.py check $*.ssnail $*.gguf $(TEXT)

%.info: %.gguf
	$(CONVERT) info $<

findmodels:
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_findmodels.py $(QUERY) $(if $(MEM),--mem $(MEM))

# The candidate list: one URL per line, the most precise file of each model
# (written once; edit it, or make modellist to search again)
$(MODEL_LIST):
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_findmodels.py $(QUERY) --urls $@ > /dev/null

modellist:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(SSNAIL_TOOLS)ssnail_findmodels.py $(QUERY) --urls $(MODEL_LIST) > /dev/null

# Fetch what's missing, then convert every GGUF in MODELS_DIR (fetched or put
# there by hand), unattended: a model or variant that won't fit, would need
# quantising below its file's precision, or would exceed the model's trained
# context is skipped with the converter's message.
allmodels: $(MODEL_LIST)
	-$(PYTHON) $(SSNAIL_TOOLS)ssnail_findmodels.py --fetch $(MODEL_LIST) --dir $(MODELS_DIR)
	-@$(MAKE) -k --no-print-directory -f $(firstword $(MAKEFILE_LIST)) allmodels-images
	@echo; echo "Images in $(MODELS_OUT)/:"; ls -1s $(MODELS_OUT)/ 2>/dev/null || echo "  (none)"

MODEL_STEMS = $(patsubst $(MODELS_DIR)/%.gguf,%,$(MODEL_GGUFS))
allmodels-images: $(MODEL_STEMS:%=$(MODELS_OUT)/%.ssnail) \
  $(foreach c,$(MODELS_CTX),$(MODEL_STEMS:%=$(MODELS_OUT)/%.ctx$(c).ssnail))

MODEL_OPTS = --wtype $(WTYPE) --fallback $(FALLBACK) --mem-mb $(MODELS_MEM) $(if $(TYPE),--type "$(TYPE)")

$(MODELS_OUT)/%.ssnail: $(MODELS_DIR)/%.gguf $(TOOL_DEPS)
	@mkdir -p $(MODELS_OUT)
	$(CONVERT) convert $< -o $@ $(MODEL_OPTS) </dev/null

define CTX_RULE
$(MODELS_OUT)/%.ctx$(1).ssnail: $(MODELS_DIR)/%.gguf $(TOOL_DEPS)
	@mkdir -p $(MODELS_OUT)
	$$(CONVERT) convert $$< -o $$@ $$(MODEL_OPTS) --ctx $(1) --within-train </dev/null
endef
$(foreach c,$(MODELS_CTX),$(eval $(call CTX_RULE,$(c))))

client: $(SSNAIL_TOOLS)client/ssnailchat.prg sdcard/SSNCHAT.D81

$(LIBC)/build/src/libmega65libc.a:
	@test -f $(LIBC)/CMakeLists.txt || git -C $(SSNAIL_TOOLS) submodule update --init mega65-libc
	cmake -DCMAKE_PREFIX_PATH=$(LLVM_MOS) -S $(LIBC) -B $(LIBC)/build > /dev/null
	$(MAKE) -C $(LIBC)/build mega65libc

$(SSNAIL_TOOLS)client/ssnailchat.prg: $(CLIENT_SRC) $(LIBC)/build/src/libmega65libc.a
	$(MOS_CC) -Os -Wall -I $(LIBC)/include -o $@ $(filter %.c,$^) $(LIBC)/build/src/libmega65libc.a

sdcard/SSNCHAT.D81: $(SSNAIL_TOOLS)client/ssnailchat.prg
	@mkdir -p sdcard
	rm -f $@
	c1541 -format "ssnail chat,sc" d81 $@ -write $< "ssnail chat" > /dev/null

$(SSNAIL_TOOLS)client/tok_host: $(SSNAIL_TOOLS)client/tok_host.c $(SSNAIL_TOOLS)client/ssnail_tokc.c
	cc -O2 -Wall -Wno-unused-result -o $@ $^

tokcheck-c: $(SSNAIL_TOOLS)client/tok_host
	$(PYTHON) $(SSNAIL_TOOLS)client/tok_check.py $(wildcard $(MODELS_OUT)/*.ssnail)

fetch:
	@test -n "$(URL)" || { echo "usage: make fetch URL=https://huggingface.co/<repo>/resolve/main/<file>.gguf"; exit 1; }
	curl -L --fail -C - -o "$(notdir $(URL))" "$(URL)"

test:
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_findmodels.py > /dev/null && echo "findmodels: ALL PASSED"
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_tokenizer.py
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_auto.py
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_hw.py
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_pipeline.py
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_gpt2_bert.py
	$(PYTHON) $(SSNAIL_TOOLS)tests/test_desc.py

clean:
	rm -f *.ssnail
