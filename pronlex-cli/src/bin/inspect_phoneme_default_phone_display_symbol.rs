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

    for (i, phone) in phonemicized.phones.iter().enumerate() {
        println!("Phone [{}]:", i);
        let phone_ipa_val = match &phone.phone {
            speech::Spec::Known(id) => id.as_str().strip_prefix("ipa.phone.").unwrap_or(id.as_str()),
            _ => "",
        };
        println!("  phone_ipa: {:?}", phone_ipa_val);

        let phoneme_id = phonemicized.phonemes.iter().find(|p| {
            p.realized_as.iter().any(|rp| {
                rp.phone == phone.phone && rp.features == phone.features && rp.span == phone.span
            })
        }).and_then(|p| match &p.phoneme {
            speech::Spec::Known(ref id) => Some(id.clone()),
            _ => None,
        });

        if let Some(pid) = phoneme_id {
            let symbol = speech::phoneme_default_phone_display_symbol(&pid, &phonemicized.variety);
            println!("  phoneme_id: {:?}", pid);
            println!("  default_phone_display_symbol: {:?}", symbol);
        } else {
            println!("  no phoneme found");
        }
    }
}
