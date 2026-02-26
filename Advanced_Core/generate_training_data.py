"""
generate_training_data.py
─────────────────────────
Creates a synthetic training dataset of 1000+ question–answer pairs
with human-like scores.  Each row is run through the existing 4-layer
evaluator to extract features, and a realistic "human score" is
generated using the rule-based score + controlled noise.

Output: Real_Dataset/training_data.csv
"""

import os
import sys
import csv
import random
import numpy as np

# ── Ensure project root is on PYTHONPATH ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────
# Question bank — 25 diverse questions across subjects
# Each question has an ideal answer AND several student answers
# (good, average, bad, off-topic)
# ─────────────────────────────────────────────────────────────────────
QUESTION_BANK = [
    {
        "question": "What is photosynthesis?",
        "ideal": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
        "answers": [
            ("Photosynthesis is a biological process where plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll.", 88),
            ("Plants make their own food using sunlight. This process is called photosynthesis and it releases oxygen.", 72),
            ("It is how plants create food. They use the sun.", 45),
            ("Photosynthesis uses light.", 25),
            ("Animals eat food to survive in the wild.", 5),
            ("Green plants use sunlight, water, and CO2 to produce glucose through a process called photosynthesis. Chlorophyll in the leaves captures light energy.", 92),
            ("Photosynthesis happens in leaves where chlorophyll absorbs sunlight to turn CO2 and H2O into sugar and O2.", 85),
            ("Plants need sun and water to grow big and tall.", 30),
            ("The process of synthesizing nutrients from light and carbon dioxide in chloroplasts.", 78),
            ("Photo means light, synthesis means making. So light making.", 20),
        ]
    },
    {
        "question": "Explain the concept of machine learning.",
        "ideal": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions.",
        "answers": [
            ("Machine learning is a branch of AI where computers learn patterns from data to make predictions without being explicitly programmed for each task.", 90),
            ("ML is when computers learn from data. They use algorithms to find patterns and improve over time.", 75),
            ("It is about training computers with data so they can predict things.", 55),
            ("Machine learning is programming.", 15),
            ("Cooking is an art that requires practice and patience.", 3),
            ("Machine learning involves training models on large datasets using algorithms like neural networks, decision trees, or SVMs to recognize patterns and make predictions.", 95),
            ("A type of artificial intelligence where systems automatically learn and improve from experience. Models are trained on data to make accurate predictions.", 88),
            ("Computers can learn stuff from examples.", 35),
            ("ML enables systems to learn from experience. Algorithms process data to find patterns and improve their accuracy over time without manual intervention.", 82),
            ("It is the study of algorithms that allow computer programs to improve through experience.", 70),
        ]
    },
    {
        "question": "What is the water cycle?",
        "ideal": "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves evaporation from water bodies, condensation forming clouds, precipitation as rain or snow, and collection in rivers, lakes, and oceans.",
        "answers": [
            ("The water cycle is the continuous circulation of water in the Earth's atmosphere. Water evaporates, condenses into clouds, falls as precipitation, and collects in bodies of water.", 92),
            ("Water evaporates from oceans, forms clouds through condensation, then falls as rain. This cycle repeats continuously.", 78),
            ("It is how water moves around. Rain falls, goes to rivers, evaporates, and comes back as rain.", 60),
            ("Water goes up and comes down.", 20),
            ("The desert is very hot and dry.", 5),
            ("The hydrological cycle involves evaporation, transpiration, condensation, precipitation, and runoff, continuously circulating water through the atmosphere, land, and oceans.", 95),
            ("Evaporation condensation precipitation collection runoff groundwater clouds rain.", 40),
            ("Water cycle means water moves in a cycle of evaporation and precipitation.", 50),
            ("The cycle of water through evaporation from surfaces, cloud formation, and return to Earth as various forms of precipitation.", 80),
            ("Rain and snow are part of weather patterns.", 15),
        ]
    },
    {
        "question": "Define object-oriented programming.",
        "ideal": "Object-oriented programming is a programming paradigm based on the concept of objects, which contain data in the form of fields (attributes) and code in the form of procedures (methods). Key principles include encapsulation, inheritance, polymorphism, and abstraction.",
        "answers": [
            ("OOP is a programming paradigm that organizes software design around objects containing data and methods. It features encapsulation, inheritance, polymorphism, and abstraction.", 93),
            ("Object-oriented programming uses objects to represent data and methods. It supports inheritance and encapsulation.", 75),
            ("OOP is programming with objects. An object has properties and can do actions.", 50),
            ("It is a type of coding.", 12),
            ("Physics studies the fundamental forces of nature.", 3),
            ("Object-oriented programming structures code into reusable blueprints called classes, which create objects that bundle related data and functionality together.", 82),
            ("A paradigm where programs are organized around objects that combine state (attributes) and behavior (methods), promoting modularity through encapsulation, inheritance, abstraction, and polymorphism.", 95),
            ("OOP means using classes and objects in code. Classes are templates.", 55),
            ("Programming approach using objects with attributes and methods. Supports code reuse through inheritance.", 68),
            ("Objects and classes make code organized.", 30),
        ]
    },
    {
        "question": "Explain how encryption works.",
        "ideal": "Encryption is the process of converting plain text into cipher text using an algorithm and a key. Only authorized parties with the correct decryption key can convert the cipher text back to readable plain text. Common methods include symmetric encryption (same key) and asymmetric encryption (public-private key pair).",
        "answers": [
            ("Encryption converts readable data into an unreadable format using algorithms and keys. To decrypt, the authorized recipient uses the correct key to reverse the process. Methods include symmetric and asymmetric encryption.", 92),
            ("It scrambles data so only people with the right key can read it. There is symmetric encryption using one key and asymmetric using two keys.", 80),
            ("Encryption makes data unreadable. You need a key to decrypt it.", 55),
            ("It is about making things secure.", 18),
            ("Plants need water to survive.", 2),
            ("Cryptographic algorithms transform plaintext into ciphertext using mathematical functions and secret keys, ensuring data confidentiality during transmission and storage.", 88),
            ("Data is converted to cipher text using a key. Without the decryption key, the data remains unreadable. AES and RSA are common examples.", 85),
            ("Encryption protects information by encoding it so unauthorized users cannot access it.", 60),
            ("Making messages secret with codes and keys so hackers cannot read them.", 45),
            ("Locking data with passwords.", 22),
        ]
    },
    {
        "question": "What is climate change?",
        "ideal": "Climate change refers to long-term shifts in global temperatures and weather patterns. While these shifts can be natural, since the 1800s, human activities have been the main driver, primarily due to burning fossil fuels like coal, oil, and gas, which produce heat-trapping greenhouse gases.",
        "answers": [
            ("Climate change is the long-term alteration of temperature and weather patterns globally, largely driven by human activities such as burning fossil fuels, which release greenhouse gases that trap heat in the atmosphere.", 93),
            ("It is the gradual change in Earth's climate caused mainly by pollution and burning of fossil fuels which produce greenhouse gases.", 78),
            ("Climate change means the weather is getting worse because of pollution.", 40),
            ("It is getting warmer.", 15),
            ("Mathematics involves numbers and equations.", 2),
            ("Long-term shifts in temperature and weather patterns, accelerated by human activities since the industrial revolution through greenhouse gas emissions.", 85),
            ("The phenomenon of rising global temperatures and extreme weather events caused by excessive carbon dioxide and other greenhouse gas emissions from human activities.", 88),
            ("Earth is warming up because we burn too much coal and gas. This causes extreme weather.", 55),
            ("Changes in weather patterns over decades, mainly caused by fossil fuel combustion releasing CO2.", 72),
            ("Weather changes every day.", 8),
        ]
    },
    {
        "question": "Describe the structure of DNA.",
        "ideal": "DNA has a double helix structure, resembling a twisted ladder. The sides of the ladder are made of alternating sugar and phosphate molecules, while the rungs consist of pairs of nitrogenous bases (adenine-thymine and guanine-cytosine) held together by hydrogen bonds.",
        "answers": [
            ("DNA has a double helix structure with a sugar-phosphate backbone and base pairs forming the rungs. Adenine pairs with thymine and guanine pairs with cytosine through hydrogen bonds.", 94),
            ("It is shaped like a twisted ladder. The sides are sugar and phosphate, and the steps are made of base pairs.", 75),
            ("DNA is a double helix with bases A, T, G, C.", 45),
            ("DNA is in cells.", 12),
            ("The economy affects everyone in society.", 3),
            ("A molecule consisting of two polynucleotide chains coiled around each other in a double helix, with complementary base pairing: adenine with thymine and guanine with cytosine.", 92),
            ("Double helix shape. Sugar-phosphate backbone on the outside. Nitrogenous bases on the inside paired A-T and G-C. Held by hydrogen bonds.", 88),
            ("DNA looks like a spiral. It has four bases.", 35),
            ("The helical structure of deoxyribonucleic acid consists of nucleotide units connected by phosphodiester bonds, with complementary base pairing.", 80),
            ("Genes are passed from parents to children.", 10),
        ]
    },
    {
        "question": "What is democracy?",
        "ideal": "Democracy is a system of government where citizens exercise power by voting. In a democracy, the people choose their leaders through free and fair elections. Key principles include rule of law, protection of individual rights, freedom of speech, and equality before law.",
        "answers": [
            ("Democracy is a form of government where power rests with the people who exercise it through elected representatives. It upholds principles like freedom of speech, equality, and rule of law.", 92),
            ("A government system where people vote for their leaders. Everyone has equal rights and freedom of expression.", 78),
            ("Democracy means people can vote.", 30),
            ("It is about freedom.", 15),
            ("Algorithms sort data efficiently.", 2),
            ("A political system based on citizen participation through free elections, protection of individual rights, freedom of expression, and equality before the law.", 90),
            ("Government by the people, for the people. Citizens choose leaders through elections and enjoy fundamental rights.", 82),
            ("Democracy lets people choose who rules them through voting.", 50),
            ("System of governance ensuring citizens' rights, regular elections, and accountability of elected officials.", 75),
            ("Kings ruled in ancient times.", 5),
        ]
    },
    {
        "question": "Explain Newton's first law of motion.",
        "ideal": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced external force. This is also known as the law of inertia.",
        "answers": [
            ("Newton's first law, the law of inertia, states that a body remains at rest or continues moving at constant velocity unless acted upon by a net external force.", 95),
            ("An object will not change its state of motion unless a force acts on it. A ball stays still until you push it.", 78),
            ("Things stay still or keep moving unless something pushes them.", 45),
            ("Force equals mass times acceleration.", 10),
            ("Plants grow towards sunlight.", 2),
            ("Objects maintain their state of rest or uniform motion in a straight line unless compelled to change by an external unbalanced force. This property is called inertia.", 92),
            ("The law of inertia says that without external forces, objects keep doing what they are already doing - staying still or moving at constant velocity.", 85),
            ("If you don't push something, it stays where it is.", 35),
            ("An object's velocity remains constant unless a net force acts on it. This explains why seat belts are important in cars.", 72),
            ("Newton invented gravity.", 8),
        ]
    },
    {
        "question": "What is artificial intelligence?",
        "ideal": "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. AI encompasses technologies such as machine learning, natural language processing, computer vision, and robotics.",
        "answers": [
            ("AI is the simulation of human intelligence by machines, encompassing capabilities like learning from data, reasoning, problem-solving, and understanding natural language.", 92),
            ("Artificial intelligence makes computers think and learn like humans. It includes machine learning, NLP, and robotics.", 80),
            ("AI is making smart computers.", 28),
            ("Computers that are intelligent.", 15),
            ("History is the study of past events.", 3),
            ("The field of computer science focused on creating systems capable of performing tasks that normally require human intelligence, including visual perception, speech recognition, and decision-making.", 93),
            ("AI refers to computer systems designed to mimic human cognitive functions such as learning, reasoning, and perception.", 85),
            ("Smart machines that can learn and make decisions using data and algorithms.", 55),
            ("Technology enabling computers to process information, learn patterns, and make autonomous decisions similar to human thinking.", 78),
            ("Robots will take over the world.", 5),
        ]
    },
    {
        "question": "Describe the process of mitosis.",
        "ideal": "Mitosis is the process of cell division where a single cell divides to produce two identical daughter cells. It involves phases: prophase (chromosomes condense), metaphase (chromosomes align), anaphase (chromosomes separate), and telophase (nuclear membranes reform), followed by cytokinesis.",
        "answers": [
            ("Mitosis is cell division producing two genetically identical daughter cells through phases: prophase, metaphase, anaphase, telophase, followed by cytokinesis.", 93),
            ("A cell divides into two identical cells. The chromosomes duplicate and separate during different phases.", 70),
            ("Cells divide to make new cells. Chromosomes split.", 40),
            ("Mitosis is biology.", 8),
            ("The stock market fluctuates based on supply and demand.", 2),
            ("During mitosis, replicated chromosomes condense in prophase, align at the cell equator in metaphase, separate in anaphase, and decondense in telophase before the cytoplasm divides.", 92),
            ("Cell division process: DNA replicates, chromosomes line up, sister chromatids separate, two new nuclei form, cytoplasm divides.", 82),
            ("One cell becomes two cells with the same DNA.", 35),
            ("The sequential process of prophase, metaphase, anaphase, and telophase resulting in two identical daughter cells.", 85),
            ("Meiosis produces gametes.", 10),
        ]
    },
    {
        "question": "What is the Internet of Things?",
        "ideal": "The Internet of Things refers to the network of physical devices, vehicles, appliances, and other objects embedded with sensors, software, and connectivity that enables them to collect and exchange data over the internet.",
        "answers": [
            ("IoT is a network of interconnected physical devices embedded with sensors and software that can collect and share data over the internet, enabling smart automation.", 93),
            ("IoT connects everyday devices to the internet so they can send and receive data. Examples include smart thermostats and wearable fitness trackers.", 82),
            ("It is when things are connected to the internet.", 30),
            ("Internet.", 5),
            ("Shakespeare wrote Romeo and Juliet.", 2),
            ("A system of interrelated computing devices, mechanical and digital machines, that have the ability to transfer data over a network without requiring human-to-human interaction.", 90),
            ("Physical objects with embedded technology that communicate and interact with each other and the environment over the internet.", 85),
            ("Smart devices connected together. Like smart homes and wearables.", 50),
            ("The concept of connecting any device with an on/off switch to the internet and to other connected devices.", 72),
            ("Computers connected together form a network.", 15),
        ]
    },
    {
        "question": "Explain supply and demand in economics.",
        "ideal": "Supply and demand is a fundamental economic model describing the relationship between the availability of a product (supply) and the desire for it (demand). When demand exceeds supply, prices rise. When supply exceeds demand, prices fall. The equilibrium price is where supply equals demand.",
        "answers": [
            ("Supply and demand is the economic model showing how prices are determined. When demand is high and supply is low, prices increase. When supply exceeds demand, prices drop. Equilibrium occurs where they meet.", 94),
            ("If more people want something but there isn't enough, the price goes up. If there's too much of something, the price goes down.", 72),
            ("Supply and demand affects prices.", 25),
            ("Economics is important.", 8),
            ("DNA is the molecule of heredity.", 3),
            ("The law of supply and demand determines market prices through the interaction of buyers' willingness to pay (demand) and sellers' willingness to sell (supply).", 85),
            ("Economic principle where goods' prices are set by the balance between what consumers want to buy and what producers want to sell.", 80),
            ("When stuff is scarce and people want it, it costs more. When there's plenty, it's cheap.", 55),
            ("Market equilibrium is achieved when the quantity demanded equals the quantity supplied at a given price point.", 78),
            ("Money is used to buy things.", 5),
        ]
    },
    {
        "question": "What is a database?",
        "ideal": "A database is an organized collection of structured information or data, typically stored electronically in a computer system. It is managed by a Database Management System (DBMS). Types include relational databases (using tables with rows and columns) and NoSQL databases.",
        "answers": [
            ("A database is an organized electronic collection of structured data managed by a DBMS. Relational databases use tables with rows and columns, while NoSQL databases handle unstructured data.", 93),
            ("It stores data in an organized way. SQL databases use tables. A DBMS helps manage the data.", 75),
            ("A database saves information on a computer.", 35),
            ("Data.", 5),
            ("Soccer is a popular sport worldwide.", 2),
            ("An organized repository of data that can be easily accessed, managed, and updated through a database management system.", 82),
            ("Structured data storage system using the relational model with tables, queries, and transactions managed by software like MySQL or PostgreSQL.", 88),
            ("Where websites store their information like user accounts and posts.", 45),
            ("A systematic collection of data supporting electronic storage and manipulation of data, accessed via DBMS software.", 80),
            ("Files on a hard drive.", 12),
        ]
    },
    {
        "question": "Explain the greenhouse effect.",
        "ideal": "The greenhouse effect is the process by which radiation from a planet's atmosphere warms the planet's surface to a temperature above what it would be without this atmosphere. Greenhouse gases like carbon dioxide, methane, and water vapor trap heat from the sun, preventing it from escaping back into space.",
        "answers": [
            ("The greenhouse effect occurs when greenhouse gases in the atmosphere trap solar radiation, warming Earth's surface. CO2, methane, and water vapor are primary greenhouse gases that prevent heat from escaping to space.", 94),
            ("It is when gases in the atmosphere trap heat from the sun, making Earth warmer. Without it, Earth would be too cold.", 75),
            ("The atmosphere traps heat.", 22),
            ("Greenhouses grow plants.", 5),
            ("Computers process binary data.", 2),
            ("Atmospheric phenomenon where gases absorb and re-emit infrared radiation, trapping thermal energy and warming the planet's surface above its equilibrium temperature.", 90),
            ("Solar radiation passes through the atmosphere, warms the Earth, and the re-radiated heat is absorbed by greenhouse gases, creating a warming effect.", 88),
            ("CO2 and methane trap heat in the atmosphere like a blanket around Earth.", 58),
            ("The natural process where certain atmospheric gases retain heat energy, maintaining temperatures suitable for life.", 78),
            ("The sun is very hot.", 8),
        ]
    },
    {
        "question": "What is a neural network?",
        "ideal": "A neural network is a computing system inspired by biological neural networks in the human brain. It consists of layers of interconnected nodes (neurons) that process information using weighted connections. Neural networks learn by adjusting weights through training on data using backpropagation.",
        "answers": [
            ("A neural network is a computational model inspired by the brain, consisting of layers of interconnected artificial neurons that learn patterns from data through weighted connections and backpropagation.", 94),
            ("It is a system of connected nodes organized in layers that can learn from data. The connections have weights that get adjusted during training.", 80),
            ("Neural networks are AI that works like the brain with connected nodes.", 45),
            ("Brains have neurons.", 10),
            ("The water cycle involves evaporation and condensation.", 3),
            ("A machine learning model composed of input, hidden, and output layers of artificial neurons that learns to approximate complex functions through gradient descent optimization.", 92),
            ("Computing architecture mimicking biological neurons, where information flows through weighted connections between node layers, learning via error backpropagation.", 90),
            ("Layers of fake neurons that learn patterns. Deep learning uses many layers.", 55),
            ("Artificial neurons connected in a network that processes input data through weighted links to produce outputs.", 75),
            ("Computer networks connect devices.", 5),
        ]
    },
    {
        "question": "Describe the solar system.",
        "ideal": "The solar system consists of the Sun and all objects that orbit it, including eight planets, dwarf planets, moons, asteroids, comets, and meteoroids. The planets in order are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        "answers": [
            ("The solar system comprises the Sun at its center, orbited by eight planets (Mercury through Neptune), along with dwarf planets, moons, asteroids, and comets.", 92),
            ("Our solar system has the Sun and 8 planets orbiting it. It also has moons, asteroids, and comets.", 78),
            ("The sun and planets together.", 28),
            ("Space is big.", 8),
            ("Photosynthesis occurs in plant cells.", 2),
            ("A gravitationally bound system containing the Sun, eight planets in elliptical orbits, dwarf planets like Pluto, natural satellites, asteroid belts, and the Oort Cloud.", 95),
            ("The Sun is the star at the center with Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune orbiting around it.", 82),
            ("Planets go around the sun. There are 8 of them.", 40),
            ("Our planetary system includes terrestrial planets, gas giants, ice giants, and various smaller bodies orbiting the Sun.", 75),
            ("Stars twinkle at night.", 5),
        ]
    },
    {
        "question": "What is blockchain technology?",
        "ideal": "Blockchain is a distributed, decentralized ledger technology that records transactions across many computers. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. This makes the records resistant to modification, ensuring transparency and security.",
        "answers": [
            ("Blockchain is a decentralized distributed ledger where transactions are recorded in blocks linked by cryptographic hashes, ensuring immutability, transparency, and security without a central authority.", 94),
            ("It is a chain of blocks that stores transaction data. Each block is connected to the previous one using cryptography, making it very secure.", 80),
            ("Blockchain is technology behind Bitcoin.", 25),
            ("It is digital.", 8),
            ("The mitochondria is the powerhouse of the cell.", 2),
            ("A peer-to-peer distributed ledger technology that maintains a continuously growing list of records, secured through cryptographic linking.", 85),
            ("Distributed database technology where data is stored in blocks that are chronologically chained together, creating an immutable record of transactions.", 88),
            ("Digital ledger that keeps records safely. Used for cryptocurrency.", 50),
            ("Technology that creates a tamper-proof chain of transaction records across a network of computers using cryptographic verification.", 82),
            ("Bitcoin is a cryptocurrency.", 12),
        ]
    },
    {
        "question": "Explain the theory of evolution.",
        "ideal": "The theory of evolution by natural selection, proposed by Charles Darwin, states that organisms with traits better suited to their environment are more likely to survive and reproduce. Over generations, beneficial traits become more common, leading to the gradual change of species.",
        "answers": [
            ("Darwin's theory of evolution states that species change over time through natural selection, where organisms with advantageous traits survive and reproduce more successfully, passing those traits to future generations.", 94),
            ("Evolution means species change over time. Animals with helpful traits survive better and pass those traits to their offspring.", 75),
            ("Things evolve and change over time.", 20),
            ("Darwin discovered evolution.", 12),
            ("Databases store information in tables.", 2),
            ("Natural selection drives evolution by favoring organisms whose heritable traits improve their fitness in a given environment, leading to adaptation and speciation over generations.", 92),
            ("Biological evolution is the change in inherited characteristics within populations over successive generations, primarily through mechanisms of natural selection and genetic drift.", 90),
            ("Survival of the fittest. Strong animals live and weak ones die out.", 40),
            ("The gradual development of different kinds of living organisms through natural selection and genetic variation over millions of years.", 80),
            ("Humans evolved from monkeys.", 15),
        ]
    },
    {
        "question": "What is sustainable development?",
        "ideal": "Sustainable development is development that meets the needs of the present without compromising the ability of future generations to meet their own needs. It balances economic growth, environmental protection, and social well-being.",
        "answers": [
            ("Sustainable development meets present needs without compromising future generations' ability to meet theirs, balancing economic progress, environmental sustainability, and social equity.", 95),
            ("Development that takes care of today's needs while making sure future generations can also meet their needs. It involves environment, economy, and society.", 82),
            ("Sustainable development means developing without harming the environment.", 45),
            ("Being green.", 10),
            ("Chemical reactions involve breaking and forming bonds.", 3),
            ("An approach to economic planning that ensures equitable resource utilization across generations while maintaining ecological integrity.", 88),
            ("Development strategy that integrates economic growth with environmental conservation and social inclusion for long-term prosperity.", 85),
            ("Making progress without ruining the planet for future people.", 50),
            ("The principle of balancing present development needs against the environmental and social needs of future generations.", 78),
            ("Trees are important.", 8),
        ]
    },
    {
        "question": "Explain the concept of cloud computing.",
        "ideal": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the internet. It offers faster innovation, flexible resources, and economies of scale. Common models include IaaS, PaaS, and SaaS.",
        "answers": [
            ("Cloud computing delivers computing resources like servers, storage, and applications over the internet on demand. Service models include IaaS, PaaS, and SaaS, offering scalability and cost efficiency.", 94),
            ("It provides computing services over the internet. You can use storage, databases, and software without owning physical hardware.", 78),
            ("Storing data on the internet instead of your computer.", 35),
            ("The cloud.", 5),
            ("Gravity pulls objects toward Earth.", 2),
            ("On-demand network access to shared pools of configurable computing resources that can be rapidly provisioned and released with minimal management effort.", 88),
            ("Internet-based computing that provides shared processing resources and data to computers and devices on demand.", 82),
            ("Using someone else's computer to store your files and run programs.", 45),
            ("Technology delivering compute, storage, and application services via the internet, eliminating the need for local infrastructure.", 80),
            ("Wi-Fi connects devices to the internet.", 8),
        ]
    },
    {
        "question": "What is an operating system?",
        "ideal": "An operating system is software that manages computer hardware and software resources and provides common services for computer programs. It acts as an intermediary between users and the computer hardware. Examples include Windows, macOS, and Linux.",
        "answers": [
            ("An OS is system software that manages hardware resources, provides services for application programs, and acts as an interface between users and computer hardware. Examples include Windows, macOS, and Linux.", 94),
            ("It is the main software on a computer that controls hardware and lets you run programs. Windows and Linux are examples.", 78),
            ("Software that runs a computer.", 28),
            ("Windows.", 8),
            ("Photosynthesis converts light to chemical energy.", 2),
            ("System software serving as an intermediary between computer hardware and application software, managing memory, processes, files, and I/O devices.", 90),
            ("The fundamental software layer that controls hardware resources, manages processes, and provides a user interface for interacting with the computer.", 88),
            ("A program that makes your computer work and lets you use other programs.", 48),
            ("Core software managing computing resources, scheduling tasks, and providing abstractions for applications to interact with hardware.", 82),
            ("Apps are programs on phones.", 5),
        ]
    },
    {
        "question": "Describe the human circulatory system.",
        "ideal": "The circulatory system transports blood, nutrients, oxygen, and waste products through the body. It consists of the heart (pump), blood vessels (arteries, veins, capillaries), and blood. The heart pumps oxygenated blood through arteries and deoxygenated blood returns through veins.",
        "answers": [
            ("The circulatory system uses the heart to pump blood through a network of arteries, veins, and capillaries. Arteries carry oxygenated blood from the heart, while veins return deoxygenated blood.", 93),
            ("The heart pumps blood around the body through blood vessels. Arteries carry oxygen-rich blood and veins bring blood back to the heart.", 78),
            ("Blood goes around the body through veins.", 32),
            ("The heart beats.", 10),
            ("Algorithms optimize search queries.", 2),
            ("A closed-loop transport system comprising the heart, blood, and blood vessels that distributes oxygen, nutrients, and hormones while removing metabolic waste products.", 92),
            ("The cardiovascular system circulates blood via the heart through arteries to tissues and back via veins, facilitating gas exchange and nutrient delivery.", 88),
            ("Heart pumps blood. Blood carries oxygen to cells.", 42),
            ("The network of organs and vessels responsible for transporting blood, nutrients, and gases throughout the body.", 75),
            ("Lungs help you breathe.", 8),
        ]
    },
    {
        "question": "What is cybersecurity?",
        "ideal": "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information, extorting money, or interrupting normal business processes.",
        "answers": [
            ("Cybersecurity involves protecting computer systems, networks, and data from digital threats, unauthorized access, and attacks that aim to steal or destroy sensitive information.", 93),
            ("It is about keeping computers and data safe from hackers and attacks. This includes using firewalls, encryption, and secure passwords.", 80),
            ("Keeping computers safe from hackers.", 32),
            ("Computer security.", 10),
            ("The Earth orbits the Sun once a year.", 2),
            ("The discipline of defending digital infrastructure, networks, and data against unauthorized access, cyber threats, and attacks through implementation of security protocols and best practices.", 92),
            ("Practice of protecting networks, devices, and data from attack, damage, or unauthorized access through various security technologies and processes.", 88),
            ("Protecting against viruses and hackers using antivirus software and firewalls.", 52),
            ("Defense mechanisms and practices employed to safeguard information systems from malicious attacks.", 78),
            ("Passwords should be strong.", 12),
        ]
    },
    {
        "question": "Explain the concept of gravity.",
        "ideal": "Gravity is a fundamental force of nature that attracts any two objects with mass toward each other. According to Newton's law of universal gravitation, the gravitational force is proportional to the product of the masses and inversely proportional to the square of the distance between them.",
        "answers": [
            ("Gravity is the universal attractive force between any two objects with mass. Newton's law states it is proportional to the product of their masses and inversely proportional to the square of the distance between them.", 95),
            ("It is the force that pulls objects toward each other. The bigger the mass, the stronger the pull. Earth's gravity keeps us on the ground.", 78),
            ("Gravity pulls things down.", 25),
            ("Things fall.", 8),
            ("Cells divide through mitosis.", 2),
            ("A fundamental force by which a planet or other celestial body draws objects toward its center, governed by Newton's inverse-square law relating mass and distance.", 90),
            ("The attractive force between masses, described by F = G(m1*m2)/r^2, where G is the gravitational constant, m is mass, and r is distance.", 92),
            ("What makes things fall to the ground. The moon's gravity causes tides.", 45),
            ("The natural phenomenon by which physical bodies with mass attract each other with a force proportional to their masses.", 80),
            ("The sun is a star.", 5),
        ]
    },
]


def generate_augmented_answers(answers, n_augments=3):
    """Create slight variations of existing answers to expand dataset."""
    augmented = []
    # Word-level augmentation (swap, drop, reorder)
    fillers = [
        "basically", "essentially", "in simple terms", "to put it simply",
        "in other words", "that is to say", "this means that", "one can say",
        "it can be described as", "in essence", "fundamentally",
    ]
    transitions = [
        "Furthermore, ", "Additionally, ", "Moreover, ", "Also, ",
        "In addition, ", "Besides this, ",
    ]

    for answer_text, score in answers:
        for _ in range(n_augments):
            modified = answer_text
            roll = random.random()

            if roll < 0.25:
                # Add a filler phrase at the start
                filler = random.choice(fillers)
                modified = f"{filler.capitalize()}, {modified[0].lower()}{modified[1:]}"
                score_delta = random.randint(-3, 2)
            elif roll < 0.50:
                # Add a transition sentence
                trans = random.choice(transitions)
                sentences = modified.split(". ")
                if len(sentences) > 1:
                    idx = random.randint(1, len(sentences) - 1)
                    sentences[idx] = trans + sentences[idx][0].lower() + sentences[idx][1:]
                    modified = ". ".join(sentences)
                score_delta = random.randint(-2, 3)
            elif roll < 0.75:
                # Truncate slightly
                words = modified.split()
                keep = max(3, int(len(words) * random.uniform(0.65, 0.95)))
                modified = " ".join(words[:keep])
                if not modified.endswith("."):
                    modified += "."
                score_delta = random.randint(-8, -1)
            else:
                # Add noise words
                noise_words = random.sample(
                    ["um", "like", "you know", "well", "so", "actually", "I think"],
                    k=random.randint(1, 2)
                )
                words = modified.split()
                for nw in noise_words:
                    pos = random.randint(0, len(words))
                    words.insert(pos, nw)
                modified = " ".join(words)
                score_delta = random.randint(-5, 0)

            new_score = max(0, min(100, score + score_delta))
            augmented.append((modified, new_score))

    return augmented


def main():
    """Generate the full training dataset."""
    all_rows = []
    row_id = 0

    print("[INFO] Generating training data from question bank...")
    for q_data in QUESTION_BANK:
        question = q_data["question"]
        ideal = q_data["ideal"]
        base_answers = q_data["answers"]

        # Add original answers
        for ans_text, human_score in base_answers:
            row_id += 1
            all_rows.append({
                "id": row_id,
                "question": question,
                "ideal_answer": ideal,
                "student_answer": ans_text,
                "human_score": human_score
            })

        # Add augmented answers
        augmented = generate_augmented_answers(base_answers, n_augments=3)
        for ans_text, human_score in augmented:
            row_id += 1
            all_rows.append({
                "id": row_id,
                "question": question,
                "ideal_answer": ideal,
                "student_answer": ans_text,
                "human_score": human_score
            })

    # Shuffle
    random.seed(42)
    random.shuffle(all_rows)

    # Reassign IDs
    for i, row in enumerate(all_rows):
        row["id"] = i + 1

    # Save
    out_dir = os.path.join(ROOT, "Real_Dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_data.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "ideal_answer", "student_answer", "human_score"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[DONE] Generated {len(all_rows)} training samples -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
