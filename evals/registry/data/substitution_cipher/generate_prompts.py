import random

random.seed(1)
alphabet = 'abcdefghijklmnopqrstuvwxyz'

texts = [
    "This royal throne of kings, this sceptred isle, This earth of majesty, this seat of Mars, This other Eden, demi-paradise, This fortress built by Nature for herself, Against infection and the hand of war",
    "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers. Where's the peck of pickled peppers Peter Piper picked?",
    "Betty Botter bought some butter. But she said the butters bitter. If I put it in my batter, it will make my batter bitter. But a bit of better butter will make my batter better. So twas better Betty Botter bought a bit of better butter.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood? He would chuck, he would, as much as he could, and chuck as much wood. As a woodchuck would if a woodchuck could chuck wood. She sells seashells by the seashore",
    "I scream, you scream, we all scream for ice cream",
    "Susie works in a shoeshine shop. Where she shines she sits, and where she sits she shines",
    "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy was not fuzzy, was he?"
    "The quick brown fox jumps over the lazy dog",
    "the quickest brown sly fox jumps over thirteen big lazy dogs",
    "All the world's a stage, and all the men and women merely players. They have their exits and their entrances; And one man in his time plays many parts.",
    "Is this a dagger which I see before me, the handle toward my hand?",
    "Love looks not with the eyes, but with the mind; and therefore is winged Cupid painted blind.",
    "Full fathom five thy father lies, of his bones are coral made. Those are pearls that were his eyes. Nothing of him that doth fade, but doth suffer a sea-change into something rich and strange.",
    "To be, or not to be: that is the question",
    "He had concluded that pigs must be able to fly in Hog Heaven.",
    "While on the first date he accidentally hit his head on the beam",
    "The father handed each child a roadmap at the beginning of the 2-day road trip and explained it was so they could find their way home",
    "He walked into the basement with the horror movie from the night before playing in his head.",
    "She wasn't sure whether to be impressed or concerned that he folded underwear in neat little packages",
    "As I walked through the dense forest, I couldn't help but feel a sense of awe at the towering trees and the way their leaves rustled in the wind, creating a soothing symphony of sounds that filled my ears.",
    "After years of hard work and sacrifice, Maria finally achieved her lifelong dream of becoming a successful author, and the sense of pride and accomplishment she felt was indescribable.",
    "As I gazed out at the vast expanse of the ocean before me, I couldn't help but feel a sense of wonder at the sheer size and power of the waves crashing against the shore.",
    "Despite the many obstacles and setbacks he faced along the way, John never gave up on his goal of becoming a doctor, and his unwavering determination ultimately paid off.",
    "As the sun began to set over the horizon, casting a warm golden glow over everything in sight, I couldn't help but feel a sense of peace and contentment wash over me.",
    "Despite the fact that he had never been to this part of the world before, Tom felt strangely at home in the bustling streets of Tokyo, with its neon lights and crowded alleyways.",
    "After months of training and preparation, Sarah was finally ready to take on the challenge of climbing Mount Everest, and the thought of standing at the top of the world filled her with a mixture of excitement and fear.",
    "As I stood on the edge of the cliff, looking out at the breathtaking vista before me, I couldn't help but feel a sense of gratitude for the beauty of nature and the wonders of the world.",
    "Despite the fact that she had been warned about the dangers of the jungle, Emily couldn't resist the allure of its lush foliage and exotic wildlife, and she set out on an adventure that would change her life forever.",
    "As the thunderstorm raged outside, with lightning flashing and rain pouring down in sheets, I huddled under a blanket and felt a sense of awe at the power of nature and the forces that govern our world.",
    "Despite the fact that he had been told that he would never be able to walk again, Jake refused to give up on his dream of running a marathon, and with sheer determination and hard work, he proved his doctors wrong.",
    "As I sat on the beach, watching the waves gently lapping at the shore and feeling the warm sun on my face, I couldn't help but feel a sense of tranquility and relaxation.",
    "Despite the fact that she had been betrayed by those closest to her, Rachel refused to let bitterness and anger consume her, and instead chose to focus on forgiveness and moving forward.",
    "As I wandered through the ancient ruins, marveling at the intricate carvings and massive stone structures that had stood the test of time, I couldn't help but feel a sense of awe at the ingenuity and creativity of our ancestors.",
    "Despite the fact that she had never been particularly athletic, Karen decided to take up running and quickly discovered a passion for the sport, and the feeling of accomplishment she experienced after each race was addictive.",
    "As the flames consumed the building, with smoke billowing into the sky and sirens blaring in the distance, I couldn't help but feel a sense of horror at the destruction and chaos unfolding before me.",
    "Despite the fact that he had been born into poverty and faced numerous obstacles in his life, Marcus refused to be defeated by his circumstances and worked tirelessly to create a better future for himself and his family.",
    "As I stood in the art museum, gazing at the masterpieces hanging on the walls and studying the brushstrokes and colors, I couldn't help but feel a sense of wonder and appreciation for the beauty of human creativity.",
    "Life is a journey, not a destination.",
    "The early bird catches the worm.",
    "Laughter is the best medicine.",
    "Beauty is in the eye of the beholder.",
    "Actions speak louder than words.",
    "Practice makes perfect.",
    "All's fair in love and war.",
    "Knowledge is power.",
    "The pen is mightier than the sword.",
    "Honesty is the best policy.",
    "Variety is the spice of life.",
    "When in Rome, do as the Romans do.",
    "Where there's smoke, there's fire.",
    "You can't judge a book by its cover.",
    "Don't put all your eggs in one basket.",
    "The grass is always greener on the other side.",
    "You can't have your cake and eat it too.",
    "When the going gets tough, the tough get going.",
    "Better late than never.",
    "If at first you don't succeed, try, try again.",
    "Despite its compact size, the computer had an impressive range of capabilities and could perform complex calculations with remarkable speed.",
    "The intricately woven tapestry depicted scenes from ancient mythology, with each thread carefully chosen and skillfully placed to create a work of art.",
    "The labyrinthine maze of hallways and staircases made it easy to get lost, and even the most seasoned traveler could find themselves disoriented and confused.",
    "The elegantly designed building featured soaring columns and sweeping arches, a testament to the timeless beauty of classical architecture.",
    "The densely populated city bustled with activity day and night, with the constant hum of traffic and the chatter of pedestrians filling the air.",
    "Avast ye landlubbers, the sea be a treacherous mistress, full of danger and adventure at every turn!",
    "Ahoy there matey, the wind be picking up and the waves be getting rough, best batten down the hatches if ye know what's good for ye.",
    "Arrr, me hearties, there be gold in them thar hills, and we be the ones to claim it!",
    "Hoist the Jolly Roger, me buckos, and let the world know that we be pirates, bold and free!",
    "Shiver me timbers, that be the biggest squid I ever did see, and it be looking right at us!",
    "Blimey, that be a fine looking ship, and I be thinking we could relieve them of their cargo without much trouble.",
    "Avast there, ye scurvy dog, lay down yer arms and surrender, or face the wrath of me cutlass!",
    "Me parrot be telling me that there be a hidden cove not far from here, where we can rest and refit before our next voyage.",
    "The sea be a cruel mistress, and many a good man has been lost to her fickle ways, but we be pirates, and we know how to survive.",
    "In physics, we study the fundamental laws that govern the behavior of matter and energy in the universe.",
    "The study of physics can help us understand everything from the behavior of subatomic particles to the structure of the universe itself.",
    "The laws of thermodynamics, which describe how energy is transformed in different processes, are some of the most fundamental laws in physics.",
    "In classical mechanics, we study the motion of objects and the forces that cause that motion.",
    "Quantum mechanics, on the other hand, describes the behavior of particles on a very small scale, such as atoms and subatomic particles.",
    "The study of relativity, both special and general, has had a profound impact on our understanding of the nature of space and time.",
    "Electromagnetism is a fundamental force in the universe, and the study of its behavior has led to many technological advances.",
    "The behavior of light, both as a wave and as a particle, is a fascinating area of study in physics.",
    "The study of particle physics has led to the discovery of many new particles and has greatly expanded our understanding of the subatomic world.",
    "The laws of physics apply everywhere in the universe, from the smallest particles to the largest structures.",
    "Astrophysics is the study of the physical properties of celestial objects, including stars, planets, and galaxies.",
    "The study of cosmology seeks to understand the origin, evolution, and structure of the universe as a whole.",
    "Quantum field theory is a theoretical framework that combines quantum mechanics and special relativity, and is used to describe the behavior of particles and fields.",
    "The discovery of the Higgs boson in 2012 was a major milestone in the study of particle physics, and helped to confirm the standard model of particle physics.",
    "The concept of entropy, which describes the disorder or randomness of a system, is a central idea in the study of thermodynamics.",
    "The laws of physics are thought to be universal, meaning that they apply the same way everywhere in the universe.",
    "The study of condensed matter physics, which focuses on the behavior of solids and liquids, has led to many technological advancements.",
    "The purple banana wore a hat and danced a jig.",
    "The cheese whispered secrets to the ham sandwich.",
    "The moon sang a lullaby to the stars.",
    "The elephant danced ballet on the tightrope.",
    "The robot played the piano with its toes.",
    "The jellyfish wore a top hat and monocle.",
    "The unicorn raced a cheetah across the rainbow.",
    "The toaster went on a walk in the park.",
    "The kangaroo jumped over the moon and landed on a cloud.",
    "The giraffe learned to play the accordion.",
    "The octopus knitted a sweater for its tentacles.",
    "Blubber kumquat jigglypop bloopity-bloop gizmo",
    "Fluffernutter cheddar quack wigglesnap flibbertigibbet",
    "Snorkel kerfuffle skedaddle hoopla doodad whatchamacallit",
    "Gobbledygook fiddle-faddle whippersnapper snollygoster nincompoop",
    "Zigzag wibble-wobble kerplunk piffle hocus-pocus",
    "Ballyhoo gobbledygook blubber noodle fandango",
    "Flapdoodle ragamuffin shenanigans hullabaloo whatchamacallit",
    "Bamboozle wackadoo skedaddle bling-bling wobbly",
    "Excursion translucent sagacious recluse pivotal",
    "Auspicious chromatic jaded malevolent superfluous",
    "Capricious myriad wistful perfunctory obsequious",
    "Disparate salient adroit omniscient effervescent",
    "Ineffable mellifluous ubiquitous lugubrious quintessential",
    "Surreptitious quixotic prodigious fastidious mellifluous",
    "Phlegmatic supercilious ubiquitous serendipitous amorphous",
    "Circumspect quintessential ubiquitous amorphous defunct",
    "A late night is belief in the interrelatedness of all things.",
    "Two-finger John brings both pleasure and pain.",
    "The light at the end of the tunnel likes to have a shower in the morning.",
    "qwertyuioasdfghjklpzxcvbnm",
    "xvckfjgldhfueqiropwmzna",
    "plmoknijbuhvygctfxrdzeswaq",
    "qwertyuiopasdfghjklzxcvbnm",
    "ghojkrlifvcmybntquwxdpzas",
    "xvcfghjklbqertypwmnzaosdui",
    "mjnbvcxzlkjhgfdsapoiuytrewq",
    "zxcvbnmlkjhgfdsaqwertyuiop",
    "qazwsxedcrfvtgbyhnujmikolp",
    "plokmijnuhbygvtfcrdxeszwaq",
    "oiuytrewqasdfghjklmnbvcxz",
]

# texts = [text.lower() for text in texts]

ciphers = [''.join(random.sample(alphabet,len(alphabet))) for i in range(len(texts))]


def encrypt(plaintext, key):
    ciphertext = ''
    
    for char in plaintext:
        if char.lower() in alphabet:
            index = alphabet.index(char.lower())
            if char.isupper():
                ciphertext += key[index].upper()
            else:
                ciphertext += key[index]
        else:
            ciphertext += char
    
    return ciphertext

print(encrypt(texts[0], ciphers[0]))
print(ciphers)

with open("/home/hamzah/documents/evals/evals/registry/data/substituion_cipher/substitution_cipher_test.jsonl", 'w') as f:

    for i in range(len(texts)):
        output_str = """{"input":[{"role":"system","content":"You are an intelligent cryptographer assistant tasked to decrypt text that has been encoded using a substitution cipher."},{"role":"user","content":"Decode the following string: '"""
        output_str += encrypt(texts[i], ciphers[i])
        output_str += """', which was encoded using the substitution cipher '"""
        output_str += ciphers[i]
        output_str += """'.  Output only the decoded text."}],"ideal":"""
        output_str += '"'
        output_str += texts[i]    
        output_str += '"'
        output_str += """}\n"""

        f.write(output_str)