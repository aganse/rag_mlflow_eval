def get_dataset_name():
    return "usmilestonedocs_civilwar_all16"


def get_url_listings():
    url_listings = [
        "https://www.archives.gov/milestone-documents/compromise-of-1850#transcript",
        "https://www.archives.gov/milestone-documents/kansas-nebraska-act#transcript",
        "https://www.archives.gov/milestone-documents/dred-scott-v-sandford#transcript",
        "https://www.archives.gov/milestone-documents/telegram-announcing-the-surrender-of-fort-sumter#transcript",
        "https://www.archives.gov/milestone-documents/homestead-act#transcript",
        "https://www.archives.gov/milestone-documents/pacific-railway-act#transcript",
        "https://www.archives.gov/milestone-documents/morrill-act#transcript",
        "https://www.archives.gov/milestone-documents/emancipation-proclamation#transcript",
        "https://www.archives.gov/milestone-documents/war-department-general-order-143#transcript",
        "https://www.archives.gov/milestone-documents/wade-davis-bill#transcript",
        "https://www.archives.gov/milestone-documents/articles-of-agreement-of-surrender#transcript",
        "https://www.archives.gov/milestone-documents/13th-amendment#transcript",
        "https://www.archives.gov/milestone-documents/check-for-the-purchase-of-alaska#transcript",
        "https://www.archives.gov/milestone-documents/fort-laramie-treaty#transcript",
        "https://www.archives.gov/milestone-documents/14th-amendment#transcript",
        "https://www.archives.gov/milestone-documents/15th-amendment#transcript",
    ]
    return url_listings


def get_eval_records():
    eval_records = [
      {
        "inputs": {
          "query": "What major measures were included in the Compromise of 1850 to address disputes over slavery in territories acquired from Mexico?"
        },
        "expectations": {
          "expected_response": "The Compromise of 1850 admitted California as a free state, organized Utah and New Mexico territories with questions of slavery left to local decision, settled Texas boundary issues with federal compensation, abolished the slave trade in Washington, D.C., and enacted a stronger Fugitive Slave Law."
        }
      },
      {
        "inputs": {
          "query": "How did the Compromise of 1850 attempt to balance Northern and Southern interests?"
        },
        "expectations": {
          "expected_facts": [
            "California was admitted as a free state.",
            "A stronger Fugitive Slave Law was enacted.",
            "Utah and New Mexico territories were organized without immediately prohibiting slavery."
          ]
        }
      },

      {
        "inputs": {
          "query": "What principle did the Kansas-Nebraska Act use to determine whether slavery would be allowed in the new territories?"
        },
        "expectations": {
          "expected_response": "The Kansas-Nebraska Act relied on popular sovereignty, allowing settlers in Kansas and Nebraska to decide for themselves whether slavery would be permitted."
        }
      },
      {
        "inputs": {
          "query": "What was the significance of the Kansas-Nebraska Act for earlier compromises on slavery?"
        },
        "expectations": {
          "expected_facts": [
            "It organized the Kansas and Nebraska territories.",
            "It used popular sovereignty to address slavery.",
            "It effectively repealed or undermined the Missouri Compromise restriction on slavery north of 36°30′."
          ]
        }
      },

      {
        "inputs": {
          "query": "According to the Dred Scott decision, could Congress prohibit slavery in the federal territories?"
        },
        "expectations": {
          "expected_response": "The Supreme Court held that Congress lacked constitutional authority to prohibit slavery in the territories, declaring the Missouri Compromise unconstitutional."
        }
      },
      {
        "inputs": {
          "query": "What were the court's key conclusions in Dred Scott v. Sandford?"
        },
        "expectations": {
          "expected_facts": [
            "People of African descent were not considered citizens under the Constitution for purposes of federal citizenship.",
            "Dred Scott therefore lacked standing to sue in federal court.",
            "Congress could not ban slavery in the territories under the Court's reasoning."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did the telegram announcing the surrender of Fort Sumter report?"
        },
        "expectations": {
          "expected_response": "The telegram reported that Fort Sumter had been surrendered after bombardment by Confederate forces and communicated the outcome of the engagement to federal authorities."
        }
      },
      {
        "inputs": {
          "query": "What key events are conveyed in the Fort Sumter surrender telegram?"
        },
        "expectations": {
          "expected_facts": [
            "Fort Sumter came under attack by Confederate forces.",
            "The fort was surrendered by Union defenders.",
            "The message served as an official communication of the surrender."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did the Homestead Act require in order for a settler to obtain ownership of public land?"
        },
        "expectations": {
          "expected_response": "The Homestead Act allowed qualified settlers to claim public land and receive title after meeting conditions such as residence, improvement of the land, and a period of occupancy."
        }
      },
      {
        "inputs": {
          "query": "What opportunities did the Homestead Act provide?"
        },
        "expectations": {
          "expected_facts": [
            "Settlers could claim parcels of federal public land.",
            "Residence and improvement requirements had to be met.",
            "Ownership could be obtained after satisfying statutory conditions."
          ]
        }
      },

      {
        "inputs": {
          "query": "What was the purpose of the Pacific Railway Act?"
        },
        "expectations": {
          "expected_response": "The Pacific Railway Act promoted construction of a transcontinental railroad by authorizing railroad companies, granting land, and providing federal support for railway development."
        }
      },
      {
        "inputs": {
          "query": "How did the Pacific Railway Act encourage railroad construction?"
        },
        "expectations": {
          "expected_facts": [
            "It authorized construction of a transcontinental rail connection.",
            "It granted public lands to support railroad companies.",
            "It provided federal financial assistance or incentives."
          ]
        }
      },

      {
        "inputs": {
          "query": "What type of educational institutions did the Morrill Act support?"
        },
        "expectations": {
          "expected_response": "The Morrill Act supported colleges focused on agriculture and the mechanic arts by providing federal land grants to the states."
        }
      },
      {
        "inputs": {
          "query": "What were the main goals of the Morrill Act?"
        },
        "expectations": {
          "expected_facts": [
            "States received land or land-grant resources from the federal government.",
            "The act promoted higher education in agriculture.",
            "The act promoted instruction in the mechanic arts and practical fields."
          ]
        }
      },

      {
        "inputs": {
          "query": "Whom did the Emancipation Proclamation declare to be free?"
        },
        "expectations": {
          "expected_response": "The Emancipation Proclamation declared enslaved people in areas then in rebellion against the United States to be free, while not immediately applying to all slaveholding areas."
        }
      },
      {
        "inputs": {
          "query": "What were two major effects or provisions of the Emancipation Proclamation?"
        },
        "expectations": {
          "expected_facts": [
            "It declared freedom for enslaved people in rebellious Confederate areas.",
            "It was issued as a wartime measure under presidential authority.",
            "It authorized or encouraged the enlistment of Black soldiers in Union service."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did War Department General Order No. 143 establish?"
        },
        "expectations": {
          "expected_response": "General Order No. 143 established the Bureau of Colored Troops to organize and manage African American military units serving the Union."
        }
      },
      {
        "inputs": {
          "query": "What role did General Order 143 play in the Union war effort?"
        },
        "expectations": {
          "expected_facts": [
            "It created the Bureau of Colored Troops.",
            "It provided a system for organizing Black military units.",
            "It facilitated African American service in the Union Army."
          ]
        }
      },

      {
        "inputs": {
          "query": "What conditions did the Wade-Davis Bill impose on former Confederate states seeking readmission?"
        },
        "expectations": {
          "expected_response": "The Wade-Davis Bill required stronger loyalty requirements and demanded that a majority of white male citizens swear allegiance to the Union before a state government could be reestablished."
        }
      },
      {
        "inputs": {
          "query": "How did the Wade-Davis Bill differ from Lincoln's more lenient reconstruction approach?"
        },
        "expectations": {
          "expected_facts": [
            "It imposed stricter loyalty requirements.",
            "It required a majority rather than a small minority to take loyalty oaths.",
            "It represented a more demanding congressional approach to Reconstruction."
          ]
        }
      },

      {
        "inputs": {
          "query": "What were the basic terms of surrender in the Articles of Agreement of Surrender at Appomattox?"
        },
        "expectations": {
          "expected_response": "The agreement required Confederate forces to lay down their arms and be paroled, while allowing officers and many soldiers to return home under specified conditions."
        }
      },
      {
        "inputs": {
          "query": "What provisions helped make the Appomattox surrender relatively conciliatory?"
        },
        "expectations": {
          "expected_facts": [
            "Confederate soldiers were paroled rather than imprisoned en masse.",
            "Officers were permitted to retain certain personal property.",
            "Troops were allowed to return home after complying with the terms."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did the 13th Amendment do regarding slavery?"
        },
        "expectations": {
          "expected_response": "The 13th Amendment abolished slavery and involuntary servitude in the United States except as punishment for a crime after lawful conviction."
        }
      },
      {
        "inputs": {
          "query": "What are the main provisions of the 13th Amendment?"
        },
        "expectations": {
          "expected_facts": [
            "Slavery was prohibited throughout the United States.",
            "Involuntary servitude was prohibited except as punishment for crime.",
            "Congress was given authority to enforce the amendment through legislation."
          ]
        }
      },

      {
        "inputs": {
          "query": "What was the purpose of the check for the purchase of Alaska?"
        },
        "expectations": {
          "expected_response": "The check documented payment by the United States to Russia for the acquisition of Alaska, completing the financial aspect of the territorial purchase."
        }
      },
      {
        "inputs": {
          "query": "What historical transaction is represented by the Alaska purchase check?"
        },
        "expectations": {
          "expected_facts": [
            "The United States purchased Alaska from Russia.",
            "The check served as payment for the acquisition.",
            "The document reflects a major territorial expansion of the United States."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did the 1868 Fort Laramie Treaty promise regarding Native lands?"
        },
        "expectations": {
          "expected_response": "The treaty recognized certain lands, including the Great Sioux Reservation, for Native use and sought to limit unauthorized intrusion by non-Native settlers."
        }
      },
      {
        "inputs": {
          "query": "What obligations and protections were included in the Fort Laramie Treaty?"
        },
        "expectations": {
          "expected_facts": [
            "Specific territory was recognized for Sioux use and occupation.",
            "Unauthorized settlement or intrusion by outsiders was restricted.",
            "The agreement sought to establish peace between the United States and Native nations."
          ]
        }
      },

      {
        "inputs": {
          "query": "How did the 14th Amendment define citizenship?"
        },
        "expectations": {
          "expected_response": "The 14th Amendment declared that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the state in which they reside."
        }
      },
      {
        "inputs": {
          "query": "What are three major constitutional principles established by the 14th Amendment?"
        },
        "expectations": {
          "expected_facts": [
            "It established birthright citizenship.",
            "It prohibited states from depriving persons of due process of law.",
            "It required states to provide equal protection of the laws."
          ]
        }
      },

      {
        "inputs": {
          "query": "What voting rights protection is provided by the 15th Amendment?"
        },
        "expectations": {
          "expected_response": "The 15th Amendment prohibits denying or abridging the right to vote on account of race, color, or previous condition of servitude."
        }
      },
      {
        "inputs": {
          "query": "How did the 15th Amendment build upon earlier Reconstruction amendments?"
        },
        "expectations": {
          "expected_facts": [
            "It followed the abolition of slavery established by the 13th Amendment.",
            "It extended protections for formerly enslaved people into the political sphere by protecting voting rights.",
            "Congress was granted enforcement authority through appropriate legislation."
          ]
        }
      },

      {
        "inputs": {
          "query": "What did the Act Establishing Yellowstone National Park do with the land it described?"
        },
        "expectations": {
          "expected_response": "The act set aside the Yellowstone region as a public park or pleasuring ground for the benefit and enjoyment of the people and placed it under federal protection."
        }
      },
      {
        "inputs": {
          "query": "What conservation principles appear in the Yellowstone National Park Act?"
        },
        "expectations": {
          "expected_facts": [
            "The land was reserved from settlement and private disposal.",
            "The area was designated for public benefit and enjoyment.",
            "Federal authorities were tasked with protecting natural features and resources."
          ]
        }
      },

      {
        "inputs": {
          "query": "What kinds of business conduct did the Sherman Anti-Trust Act seek to prevent?"
        },
        "expectations": {
          "expected_response": "The Sherman Anti-Trust Act sought to prevent restraints of trade and monopolistic practices that interfered with competitive commerce."
        }
      },
      {
        "inputs": {
          "query": "What are the central provisions of the Sherman Anti-Trust Act?"
        },
        "expectations": {
          "expected_facts": [
            "Contracts, combinations, or conspiracies in restraint of trade were prohibited.",
            "Monopolization or attempts to monopolize were made unlawful.",
            "Federal enforcement mechanisms were authorized."
          ]
        }
      }
    ]
    return eval_records
