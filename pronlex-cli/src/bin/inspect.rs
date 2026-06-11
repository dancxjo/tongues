use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

fn main() {
    let phonemicizer = EnglishPhonemicizer;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: "zenor".to_string(),
            variety: VarietyId("en-US".to_string()),
            style: None,
        })
        .unwrap();

    println!("Syllables:");
    for (i, s) in phonemicized.syllables.iter().enumerate() {
        println!("  Syllable [{}]: stress: {:?}", i, s.stress);
        for (j, p) in s.phones.iter().enumerate() {
            println!("    Phone [{}]: phone: {:?}", j, p.phone);
        }
    }
}
