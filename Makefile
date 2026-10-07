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
#   make fetch URL=https://huggingface.co/.../resolve/main/foo.gguf
#   make test
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

CONVERT_OPTS  = --wtype $(WTYPE) --fallback $(FALLBACK) --mem-mb $(ATTICRAM)
CONVERT_OPTS += $(if $(CONTEXT_WINDOW),--ctx $(CONTEXT_WINDOW))

.PHONY: test clean findmodels fetch %.all %.chat %.run %.tokcheck
.PRECIOUS: %.ssnail

%.ssnail: %.gguf $(TOOL_DEPS)
	$(CONVERT) convert $< -o $@ $(CONVERT_OPTS)

SIZED_OPTS = --wtype $(WTYPE) --fallback $(FALLBACK) $(if $(CONTEXT_WINDOW),--ctx $(CONTEXT_WINDOW))

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

clean:
	rm -f *.ssnail
