import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-biobert-base-msmarco")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# dummy_query = [
#     'Rutgers is a good university. I like my experience there.',
#     "Hello, my dog is cute. My cute dog is amazing.",
#     'Florida is a nice place but tiger king may be better',
# ]
#
# dummy_passage = [
#     'My cat is really cute but my dog is even better.',
#     'My cat is really cute but my dog is even better.',
#     'My cat is really cute but my dog is even better.',
# ]
#
# model.eval()
# with torch.no_grad():
#     for idx in range(len(dummy_query)):
#         input_ids = torch.tensor(tokenizer.encode(text=dummy_query[idx], \
#             text_pair=dummy_passage[idx], add_special_tokens=True)).unsqueeze(0)
#         outputs = model(input_ids)
#         print(outputs)

dummy_query = "When did Beyonce start becoming popular?"

dummy_passages = [
    "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".",

    "Following the disbandment of Destiny's Child in June 2005, she released her second solo album, B'Day (2006), which contained hits \"Déjà Vu\", \"Irreplaceable\", and \"Beautiful Liar\". Beyoncé also ventured into acting, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006) and Obsessed (2009). Her marriage to rapper Jay Z and portrayal of Etta James in Cadillac Records (2008) influenced her third album, I Am... Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce and earned a record-setting six Grammy Awards in 2010, including Song of the Year for \"Single Ladies (Put a Ring on It)\". Beyoncé took a hiatus from music in 2010 and took over management of her career; her fourth album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul. Her critically acclaimed fifth studio album, Beyoncé (2013), was distinguished from previous releases by its experimental production and exploration of darker themes.",

    "The terms upper case and lower case can be written as two consecutive words, connected with a hyphen (upper-case and lower-case), or as a single word (uppercase and lowercase). These terms originated from the common layouts of the shallow drawers called type cases used to hold the movable type for letterpress printing. Traditionally, the capital letters were stored in a separate case that was located above the case that held the small letters, and the name proved easy to remember since capital letters are taller.",

    "The scientific evidence is mixed as to whether companionship of a dog can enhance human physical health and psychological wellbeing. Studies suggesting that there are benefits to physical health and psychological wellbeing have been criticised for being poorly controlled, and finding that \"[t]he health of elderly people is related to their health habits and social supports but not to their ownership of, or attachment to, a companion animal.\" Earlier studies have shown that people who keep pet dogs or cats exhibit better mental and physical health than those who do not, making fewer visits to the doctor and being less likely to be on medication than non-guardians.",

    "Kanye Omari West (/ˈkɑːnjeɪ/; born June 8, 1977) is an American hip hop recording artist, record producer, rapper, fashion designer, and entrepreneur. He is among the most acclaimed musicians of the 21st century, attracting both praise and controversy for his work and his outspoken public persona.",
]

model.eval()
with torch.no_grad():
    for idx in range(len(dummy_passages)):
        input_ids = torch.tensor(tokenizer.encode(text=dummy_query,
            text_pair=dummy_passages[idx], add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        print(outputs)
