use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;
use sudachi::analysis::stateful_tokenizer::StatefulTokenizer as Tokenizer;
use sudachi::config::Config;
use sudachi::dic::dictionary::JapaneseDictionary;
use sudachi::prelude::*;
use wana_kana::ConvertJapanese;

pub struct SudachiAnalyzer {
    dictionary: JapaneseDictionary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WordType {
    Noun,
    Verb,
}

pub struct MorphemeData {
    pub surface: String,
    pub dictionary_form: String,
    pub part_of_speech: String,
    pub reading: String,
}

#[derive(Clone, Debug)]
pub struct WordData {
    pub count: usize,
    pub reading: String,
    pub pos: String,
    pub is_sahen: bool,
}

impl SudachiAnalyzer {
    pub fn load_dictionary() -> Self {
        println!("辞書を読み込んでいます... (少し時間がかかります)");
        let dict_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources").join("system_full.dic");

        let config = Config::new(None, None, Some(dict_path)).expect("Configの作成に失敗しました");
        let dictionary = JapaneseDictionary::from_cfg(&config).expect("辞書の読み込みに失敗しました");

        Self { dictionary }
    }

    pub fn check_noun_validity(&self, text: &str) -> bool {
        self.check_validity(text, WordType::Noun)
    }

    pub fn check_verb_validity(&self, text: &str) -> bool {
        self.check_validity(text, WordType::Verb)
    }

    fn check_validity(&self, text: &str, word_type: WordType) -> bool {
        let mut tokenizer = Tokenizer::new(&self.dictionary, Mode::C);
        let mut morphemes = MorphemeList::empty(&self.dictionary);

        tokenizer.reset().push_str(text);
        let mut is_valid = false;

        if tokenizer.do_tokenize().is_ok() && morphemes.collect_results(&mut tokenizer).is_ok() {
            for m in morphemes.iter() {
                let pos = m.part_of_speech();
                match word_type {
                    WordType::Noun => {
                        if &*m.surface() == text && pos[0] == "名詞" {
                            is_valid = true;
                        }
                    }
                    WordType::Verb => {
                        let is_verb = pos[0] == "動詞";
                        let is_sahen_noun = pos[0] == "名詞" && pos.iter().any(|s| s == "サ変可能");
                        if (is_verb || is_sahen_noun) && &*m.dictionary_form() == text {
                            is_valid = true;
                        }
                    }
                }
            }
        }
        is_valid
    }

    pub fn get_word_info(&self, text: &str) -> (String, Vec<MorphemeData>, String) {
        let mut tokenizer = Tokenizer::new(&self.dictionary, Mode::C);
        let mut sudachi_morphemes = MorphemeList::empty(&self.dictionary);

        tokenizer.reset().push_str(text);
        let mut reading_full = String::new();
        let mut morpheme_data_list = Vec::new();

        if tokenizer.do_tokenize().is_ok() && sudachi_morphemes.collect_results(&mut tokenizer).is_ok() {
            for m in sudachi_morphemes.iter() {
                let surface = m.surface().to_string();
                let dict = m.dictionary_form().to_string();
                let pos = m.part_of_speech().join(",");
                let reading = m.reading_form().to_string();
                reading_full.push_str(&reading);
                morpheme_data_list.push(MorphemeData {
                    surface,
                    dictionary_form: dict,
                    part_of_speech: pos,
                    reading,
                });
            }
        }

        let mut romaji = reading_full.to_romaji();
        if romaji.is_empty() {
            romaji = format!("word_{}", text.chars().map(|c| format!("{:x}", c as u32)).collect::<String>());
        }

        let morph_str = morpheme_data_list.iter()
            .map(|m| format!("{}\t{}\t{}\t{}", m.surface, m.dictionary_form, m.part_of_speech, m.reading))
            .collect::<Vec<_>>()
            .join(" ; ");

        (romaji, morpheme_data_list, morph_str)
    }

    pub fn extract_collocations(&self, file_path: &Path, target_word: &str, target_type: WordType) -> Vec<(String, WordData)> {
        let file = File::open(file_path).expect("ファイルが開けませんでした");
        let mmap = unsafe { Mmap::map(&file).expect("メモリマップに失敗しました") };
        let content = std::str::from_utf8(&mmap).expect("ファイルが正しいUTF-8ではありません");

        let finder_target = memchr::memmem::Finder::new(target_word.as_bytes());
        let finder_wo = memchr::memmem::Finder::new("を".as_bytes());

        let result_map = content
            .par_lines()
            .fold(
                HashMap::<String, WordData>::new,
                |mut local_map, line| {
                    if finder_target.find(line.as_bytes()).is_none() || finder_wo.find(line.as_bytes()).is_none() {
                        return local_map;
                    }

                    let mut tokenizer = Tokenizer::new(&self.dictionary, Mode::C);
                    let mut morphemes = MorphemeList::empty(&self.dictionary);

                    tokenizer.reset().push_str(line);
                    if tokenizer.do_tokenize().is_ok() && morphemes.collect_results(&mut tokenizer).is_ok() {
                        let len = morphemes.len();
                        for i in 0..len.saturating_sub(2) {
                            let m1 = morphemes.get(i);
                            let m2 = morphemes.get(i + 1);
                            let m3 = morphemes.get(i + 2);

                            if &*m2.surface() == "を" && m2.part_of_speech()[0] == "助詞" {
                                match target_type {
                                    WordType::Noun => {
                                        if &*m1.surface() == target_word && m1.part_of_speech()[0] == "名詞" {
                                            let p3 = m3.part_of_speech();
                                            let is_sahen = p3.iter().any(|s| s == "サ変可能");
                                            if p3[0] == "動詞" || is_sahen {
                                                let word = &*m3.dictionary_form();

                                                let mut sub_tokenizer = Tokenizer::new(&self.dictionary, Mode::C);
                                                let mut sub_morphemes = MorphemeList::empty(&self.dictionary);
                                                sub_tokenizer.reset().push_str(word);
                                                let mut base_reading = String::new();

                                                if sub_tokenizer.do_tokenize().is_ok() && sub_morphemes.collect_results(&mut sub_tokenizer).is_ok() {
                                                    for sub_m in sub_morphemes.iter() {
                                                        base_reading.push_str(&sub_m.reading_form());
                                                    }
                                                }

                                                let data = local_map.entry(word.to_string()).or_insert(WordData {
                                                    count: 0,
                                                    reading: base_reading,
                                                    pos: if is_sahen { String::from("サ変可能名詞") } else { p3[0].clone() },
                                                    is_sahen,
                                                });
                                                data.count += 1;
                                            }
                                        }
                                    }
                                    WordType::Verb => {
                                        if &*m3.dictionary_form() == target_word {
                                            let p1 = m1.part_of_speech();
                                            if p1[0] == "名詞" && !["非自立", "代名詞", "数詞"].contains(&p1[1].as_str()) {
                                                let word = &*m1.surface();
                                                let is_sahen = p1.iter().any(|s| s == "サ変可能");
                                                let data = local_map.entry(word.to_string()).or_insert(WordData {
                                                    count: 0,
                                                    reading: m1.reading_form().to_string(),
                                                    pos: if is_sahen { String::from("サ変可能名詞") } else { p1[0].clone() },
                                                    is_sahen: false,
                                                });
                                                data.count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    local_map
                },
            )
            .reduce(
                HashMap::<String, WordData>::new,
                |mut map1, map2| {
                    for (k, v) in map2 {
                        if let Some(data) = map1.get_mut(&k) {
                            data.count += v.count;
                        } else {
                            map1.insert(k, v);
                        }
                    }
                    map1
                },
            );

        let mut sorted: Vec<_> = result_map.into_iter().collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        sorted
    }

    pub fn export_to_markdown(&self, target_word: &str, word_type: WordType, results: &[(String, WordData)], base_dir: &Path) {
        let sub_dir = match word_type {
            WordType::Noun => "nouns",
            WordType::Verb => "verbs",
        };

        let out_dir = base_dir.join(sub_dir);

        let (romaji, morphemes, morph_str) = self.get_word_info(target_word);
        let out_file = out_dir.join(format!("{}.md", romaji));
        let mut file = File::create(&out_file).expect("出力ファイルの作成に失敗しました");

        let is_target_sahen = morphemes.first().map_or(false, |m| m.part_of_speech.contains("サ変可能"));
        let display_target = if word_type == WordType::Verb && is_target_sahen {
            format!("{}する", target_word)
        } else {
            target_word.to_string()
        };

        writeln!(file, "+++").ok();
        writeln!(file, "title = \"{}\"", display_target).ok();
        writeln!(file, "morph = \"{}\"", morph_str).ok();
        writeln!(file, "weight = 0").ok();
        writeln!(file, "+++").ok();
        writeln!(file, "\n## 形態素解析").ok();

        writeln!(file, "| 表層形 | 辞書形 | 品詞 | 読み |").ok();
        writeln!(file, "|---|---|---|---|").ok();
        for m in morphemes {
            writeln!(file, "| {} | {} | {} | {} |", m.surface, m.dictionary_form, m.part_of_speech, m.reading).ok();
        }

        match word_type {
            WordType::Noun => {
                writeln!(file, "\n## {}を〇〇", display_target).ok();
            }
            WordType::Verb => {
                writeln!(file, "\n## 〇〇を{}", display_target).ok();
            }
        }

        writeln!(file, "| 単語 | 出現回数 | 読み（ローマ字） | 品詞 |\n|---|---|---|---|").ok();

        for (word, data) in results {
            let display_word = if data.is_sahen {
                format!("{}する", word)
            } else {
                word.to_string()
            };

            let mut romaji = data.reading.to_romaji();
            if romaji.is_empty() {
                continue;
            }

            writeln!(file, "| {} | {} | {} | {} |", display_word, data.count, romaji, data.pos).ok();
        }
        println!("結果を {} に保存しました", out_file.display());
    }

    pub fn process_directory_links_and_weights(&self, base_dir: &Path) {
        let nouns_dir = base_dir.join("nouns");
        let verbs_dir = base_dir.join("verbs");

        let mut noun_romajis = std::collections::HashSet::new();
        let mut verb_romajis = std::collections::HashSet::new();

        if let Ok(entries) = std::fs::read_dir(&nouns_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".md") && name != "_index.md" { noun_romajis.insert(name.trim_end_matches(".md").to_string()); }
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&verbs_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".md") && name != "_index.md" { verb_romajis.insert(name.trim_end_matches(".md").to_string()); }
                }
            }
        }

        let process_dir = |dir: &Path, target_romajis: &std::collections::HashSet<String>, link_prefix: &str| {
            let mut files_data = Vec::new();
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                        if filename.ends_with(".md") && filename != "_index.md" {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                let mut reading = String::new();
                                for line in content.lines() {
                                    if line.starts_with("morph = \"") {
                                        if let Some(s) = line.split('"').nth(1) {
                                            for p in s.split(" ; ") {
                                                if let Some(r) = p.split('\t').last() { reading.push_str(r); }
                                            }
                                        }
                                        break;
                                    }
                                }
                                files_data.push((path, reading, content));
                            }
                        }
                    }
                }
            }

            files_data.sort_by(|a, b| a.1.cmp(&b.1));

            for (i, (path, _, content)) in files_data.iter().enumerate() {
                let weight = i + 1;
                let mut new_content = String::with_capacity(content.len());
                let mut frontmatter_count = 0;
                let mut in_table = false;

                for line in content.lines() {
                    if line == "+++" {
                        frontmatter_count += 1;
                    }
                    if line.starts_with("weight = ") && frontmatter_count == 1 {
                        new_content.push_str(&format!("weight = {}\n", weight));
                        continue;
                    }

                    if line.starts_with("| 単語 | 出現回数 |") {
                        in_table = true;
                        new_content.push_str(line);
                        new_content.push('\n');
                        continue;
                    }
                    if in_table && line.starts_with("|---|---|") {
                        new_content.push_str(line);
                        new_content.push('\n');
                        continue;
                    }
                    if in_table && line.starts_with('|') {
                        let parts: Vec<&str> = line.split('|').collect();
                        if parts.len() >= 3 {
                            let w_col = parts[1].trim();
                            let count_col = parts[2].trim();
                            let reading_col = parts.get(3).map_or("", |s| s.trim());
                            let pos_col = parts.get(4).map_or("", |s| s.trim());

                            let mut word = w_col.to_string();
                            if word.starts_with('[') && word.contains("](") {
                                word = word.split("](").next().unwrap().trim_start_matches('[').to_string();
                            }

                            if target_romajis.contains(reading_col) {
                                new_content.push_str(&format!("| [{}]({}/{}.md) | {} | {} | {} |\n", word, link_prefix, reading_col, count_col, reading_col, pos_col));
                            } else {
                                new_content.push_str(&format!("| {} | {} | {} | {} |\n", word, count_col, reading_col, pos_col));
                            }
                            continue;
                        }
                    }
                    new_content.push_str(line);
                    new_content.push('\n');
                }
                let _ = std::fs::write(path, new_content);
            }
        };

        process_dir(&nouns_dir, &verb_romajis, "@/verbs");
        process_dir(&verbs_dir, &noun_romajis, "@/nouns");
    }
}

fn main() {
    println!("📦 辞書を読み込んでいます...");
    let analyzer = SudachiAnalyzer::load_dictionary();
    println!("\n準備完了！");
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Read Error");
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let command = parts[0];
        if command == "exit" {
            break;
        }

        match command {
            "verbs" => {
                if parts.len() < 4 {
                    println!("使い方: verbs <入力ファイル> <名詞> <出力ディレクトリ>");
                    continue;
                }

                let input_file = parts[1];
                let target_word = parts[2];
                let output_dir = parts[3];

                if !Path::new(input_file).exists() {
                    println!("ファイル '{}' が見つかりませんでした", input_file);
                    continue;
                }

                if !Path::new(output_dir).exists() {
                    println!("フォルダ '{}' が見つかりませんでした", output_dir);
                    continue;
                }

                if !analyzer.check_noun_validity(target_word) {
                    println!("'{}' は名詞として認識できませんでした", target_word);
                    continue;
                }

                let start = std::time::Instant::now();
                let results = analyzer.extract_collocations(Path::new(input_file), target_word, WordType::Noun);
                analyzer.export_to_markdown(target_word, WordType::Noun, &results, Path::new(output_dir));
                println!("処理時間: {:?}", start.elapsed());
            }
            "nouns" => {
                if parts.len() < 4 {
                    println!("❌ 使い方: nouns <入力ファイル> <動詞> <出力ディレクトリ>");
                    continue;
                }

                let input_file = parts[1];
                let target_word = parts[2];
                let output_dir = parts[3];

                if !Path::new(input_file).exists() {
                    println!("ファイル '{}' が見つかりませんでした", input_file);
                    continue;
                }

                if !Path::new(output_dir).exists() {
                    println!("フォルダ '{}' が見つかりませんでした", output_dir);
                    continue;
                }

                if !analyzer.check_verb_validity(target_word) {
                    println!("'{}' は動詞として認識できませんでした", target_word);
                    continue;
                }

                let start = std::time::Instant::now();
                let results = analyzer.extract_collocations(Path::new(input_file), target_word, WordType::Verb);
                analyzer.export_to_markdown(target_word, WordType::Verb, &results, Path::new(output_dir));
                println!("処理時間: {:?}", start.elapsed());
            }
            "relink" => {

                if parts.len() < 2 {
                    println!("使い方: relink <出力ディレクトリ>");
                    continue;
                }

                let output_dir = parts[1];
                let base_dir = Path::new(output_dir);

                if !base_dir.exists() {
                    println!("フォルダ '{}' が見つかりませんでした", output_dir);
                    continue;
                }

                let nouns_dir = base_dir.join("nouns");
                let verbs_dir = base_dir.join("verbs");

                if  !nouns_dir.exists() || !verbs_dir.exists() {
                    println!("nounsフォルダとverbsフォルダが見つかりませんでした");
                    continue;
                }

                let start = std::time::Instant::now();
                analyzer.process_directory_links_and_weights(base_dir);
                println!("処理時間: {:?}", start.elapsed());
            }
            "search" => {
                if parts.len() < 3 {
                    println!("使い方: search <入力ファイル> <検索文字列>");
                    continue;
                }

                let input_file = parts[1];
                let query = parts[2];

                if !Path::new(input_file).exists() {
                    println!("入力ファイル '{}' が見つかりませんでした", input_file);
                    continue;
                }

                let Ok(file) = File::open(input_file) else {
                    println!("ファイルを開けませんでした");
                    continue;
                };

                let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
                    println!("メモリマップに失敗しました");
                    continue;
                };

                let Ok(content) = std::str::from_utf8(&mmap) else {
                    println!("ファイルが正しいUTF-8ではありません");
                    continue;
                };

                let start = std::time::Instant::now();
                content.par_lines().filter(|line| line.contains(query)).for_each(|line| println!("{}", line));
                println!("処理時間: {:?}", start.elapsed());
            }
            _ => println!("不正なコマンドです: '{}'", command),
        }
    }
}
