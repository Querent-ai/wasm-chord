//! Tokenizer implementation for text encoding/decoding
//!
//! Supports BPE (Byte-Pair Encoding) tokenization with vocabulary loaded from GGUF metadata.

use crate::error::{Error, Result};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// Special tokens used by the tokenizer
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: String,
    /// End of sequence token
    pub eos_token: String,
    /// Padding token
    pub pad_token: Option<String>,
    /// Unknown token
    pub unk_token: String,

    /// Token IDs
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: None,
            unk_token: "<unk>".to_string(),
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
            unk_token_id: 0,
        }
    }
}

/// BPE Tokenizer with vocabulary
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Vocabulary: token string -> token ID
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: token ID -> token string
    reverse_vocab: HashMap<u32, String>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// BPE merges (pair -> merged token)
    #[allow(dead_code)] // Will be used for full BPE implementation
    merges: Vec<(String, String)>,
    /// Whether to add BOS token
    add_bos_token: bool,
    /// Whether to add EOS token
    add_eos_token: bool,
}

impl Tokenizer {
    /// Create a new tokenizer with vocabulary
    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
    ) -> Self {
        let reverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
            special_tokens,
            merges,
            add_bos_token: true,
            add_eos_token: false,
        }
    }

    /// Create a tokenizer from GGUF metadata
    pub fn from_gguf_metadata(metadata: &HashMap<String, String>) -> Result<Self> {
        // Extract vocabulary from metadata
        let vocab_size = metadata
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| Error::ParseError("Missing vocab_size in metadata".to_string()))?;

        let mut vocab = HashMap::new();
        for i in 0..vocab_size {
            let key = format!("tokenizer.ggml.token.{}", i);
            if let Some(token) = metadata.get(&key) {
                vocab.insert(token.clone(), i as u32);
            }
        }

        if vocab.is_empty() {
            return Err(Error::ParseError("No vocabulary found in metadata".to_string()));
        }

        // Extract special tokens (with fallback defaults)
        let special_tokens = SpecialTokens {
            bos_token: metadata.get("tokenizer.ggml.bos_token").cloned().unwrap_or_else(|| "<s>".to_string()),
            eos_token: metadata.get("tokenizer.ggml.eos_token").cloned().unwrap_or_else(|| "</s>".to_string()),
            unk_token: metadata.get("tokenizer.ggml.unk_token").cloned().unwrap_or_else(|| "<unk>".to_string()),
            pad_token: metadata.get("tokenizer.ggml.pad_token").cloned(),
            bos_token_id: metadata.get("tokenizer.ggml.bos_token_id")
                .and_then(|s| s.parse().ok()).unwrap_or(1),
            eos_token_id: metadata.get("tokenizer.ggml.eos_token_id")
                .and_then(|s| s.parse().ok()).unwrap_or(2),
            unk_token_id: metadata.get("tokenizer.ggml.unk_token_id")
                .and_then(|s| s.parse().ok()).unwrap_or(0),
            pad_token_id: metadata.get("tokenizer.ggml.pad_token_id")
                .and_then(|s| s.parse().ok()),
        };

        // BPE merges (if available)
        let merges = Vec::new(); // TODO: Parse from metadata if available

        Ok(Self::new(vocab, merges, special_tokens))
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize text (NFC normalization)
        let normalized: String = text.nfc().collect();

        // Tokenize using simple whitespace + vocab lookup for now
        // TODO: Implement full BPE algorithm
        let mut tokens = Vec::new();

        if add_special_tokens && self.add_bos_token {
            tokens.push(self.special_tokens.bos_token_id);
        }

        // Simple word-level tokenization (placeholder for BPE)
        for word in normalized.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // Try to find token by prefix matching
                let mut found = false;
                for len in (1..=word.len()).rev() {
                    if let Some(&token_id) = self.vocab.get(&word[..len]) {
                        tokens.push(token_id);
                        found = true;
                        break;
                    }
                }
                if !found {
                    tokens.push(self.special_tokens.unk_token_id);
                }
            }
        }

        if add_special_tokens && self.add_eos_token {
            tokens.push(self.special_tokens.eos_token_id);
        }

        Ok(tokens)
    }

    /// Decode token IDs into text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();
        let special_ids = if skip_special_tokens {
            vec![
                self.special_tokens.bos_token_id,
                self.special_tokens.eos_token_id,
                self.special_tokens.unk_token_id,
            ]
        } else {
            Vec::new()
        };

        for &token_id in token_ids {
            if skip_special_tokens && special_ids.contains(&token_id) {
                continue;
            }

            if let Some(token_str) = self.reverse_vocab.get(&token_id) {
                // Add space before token (simple joining strategy)
                if !result.is_empty() && !token_str.starts_with('<') {
                    result.push(' ');
                }
                result.push_str(token_str);
            } else {
                result.push_str(&self.special_tokens.unk_token);
            }
        }

        Ok(result.trim().to_string())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID for a specific token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token string for a specific ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Set whether to add BOS token during encoding
    pub fn set_add_bos_token(&mut self, add: bool) {
        self.add_bos_token = add;
    }

    /// Set whether to add EOS token during encoding
    pub fn set_add_eos_token(&mut self, add: bool) {
        self.add_eos_token = add;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tokenizer() -> Tokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 10);
        vocab.insert("world".to_string(), 11);
        vocab.insert("test".to_string(), 12);
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 0);

        Tokenizer::new(vocab, Vec::new(), SpecialTokens::default())
    }

    #[test]
    fn test_encode_simple() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("hello world", true).unwrap();

        // Should have BOS + hello + world
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], 1); // BOS
        assert!(tokens.contains(&10)); // hello
        assert!(tokens.contains(&11)); // world
    }

    #[test]
    fn test_decode_simple() {
        let tokenizer = create_test_tokenizer();
        let text = tokenizer.decode(&[10, 11], false).unwrap();

        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("hello", true).unwrap();

        // Should start with BOS
        assert_eq!(tokens[0], 1);
    }

    #[test]
    fn test_unknown_token() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("unknown", true).unwrap();

        // Should contain UNK token
        assert!(tokens.contains(&0));
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = create_test_tokenizer();
        assert_eq!(tokenizer.vocab_size(), 6);
    }

    #[test]
    fn test_token_lookup() {
        let tokenizer = create_test_tokenizer();

        assert_eq!(tokenizer.token_to_id("hello"), Some(10));
        assert_eq!(tokenizer.id_to_token(10), Some("hello"));
        assert_eq!(tokenizer.token_to_id("nonexistent"), None);
    }
}
