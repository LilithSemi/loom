import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('hfToGgufName maps the llama weight names', () {
    expect(hfToGgufName('model.embed_tokens.weight'), 'token_embd.weight');
    expect(hfToGgufName('model.norm.weight'), 'output_norm.weight');
    expect(hfToGgufName('lm_head.weight'), 'output.weight');
    expect(
      hfToGgufName('model.layers.0.self_attn.q_proj.weight'),
      'blk.0.attn_q.weight',
    );
    expect(
      hfToGgufName('model.layers.7.self_attn.o_proj.weight'),
      'blk.7.attn_output.weight',
    );
    expect(
      hfToGgufName('model.layers.3.post_attention_layernorm.weight'),
      'blk.3.ffn_norm.weight',
    );
    expect(
      hfToGgufName('model.layers.12.mlp.down_proj.weight'),
      'blk.12.ffn_down.weight',
    );
    expect(
      hfToGgufName('model.layers.1.input_layernorm.weight'),
      'blk.1.attn_norm.weight',
    );
  });

  test('hfToGgufName throws on an unmappable name', () {
    expect(() => hfToGgufName('model.something.unknown'), throwsArgumentError);
  });
}
