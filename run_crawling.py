from configuration.config import *
from utils.hyperparameters import *
from rl.crawler_env_tree import *
from rl.agent import *
from models.abcmodel import KwBiLSTM
from models.qnetwork import *
from rl.replay_buffer import *
from crawling.webpage import *
from crawling.textReprGenerator import *
from configuration.taxonomy import taxonomy_keywords, taxonomy_phrases

import os
import tensorflow as tf

if __name__ == "__main__":

    # GPU configuration
    # -----------------
    if GPU_AVAILABLE:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        if gpu:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpu[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU available")

    # Running
    # -------
    import gc
    import pickle
    path = "./files/"

    # Read seeds.txt file that contains the URLs of initial seeds
    f = open(f"{path}seeds.txt", "r")
    seed_urls = "".join(f.readlines()).split('\n')
    f.close()

    print("\nInitial seed docs:", len(seed_urls), "\n")
    print(seed_urls, '\n')

    [ print(f"{i}: {url}") for i,url in enumerate(seed_urls) ]

    # Classifier KwBiLSTM and CrawlerSys
    keyword_filter = KeywordFilter(taxonomy_keywords=taxonomy_keywords, new_keywords=new_keywords,
                                taxonomy_phrases=taxonomy_phrases)
    trg = TextReprGenerator(keyword_filter=keyword_filter)
    clf = KwBiLSTM(input_dim=WORD_DIM, shortcut_dim1=SHORTCUT1, shortcut_dim2=3)
    clf.load_model()

    crawler_sys = CrawlerSys(keyword_filter=keyword_filter, clf=clf)

    # Initialize the crawler environment
    env = TreeCrawlerEnv(seed_urls=seed_urls, crawler_sys=crawler_sys, TOTAL_TIME_STEPS=TOTAL_TIME_STEPS)

    # Initialize agent and reset environment
    q_network = ActionScorerBaseline()
    target_q_network = ActionScorerBaseline()
    agent = TreeDDQNAgent(env=env, q_network=q_network, target_q_network=target_q_network, 
                        target_update_period=TARGET_UPDATE_PERIOD)
    agent.initialize()

    # Env reset
    seed_webpages = agent.env.reset()       
    print(len(seed_webpages))
    print()

    # Store seed experiences to replay buffer
    seed_exps, seed_pages = agent.env.create_initial_state_actions(seed_urls)
    for i,exp in enumerate(seed_exps):
        agent.buffer.insert( exp )
        agent.env.crawling_history_ids[ seed_pages[i].id ] = seed_pages[i]

    # Initialize Tree Frontier
    agent.env.tree_frontier.initialize(initial_exp_samples=seed_exps, initial_frontier_samples=seed_webpages)

    print(len(seed_webpages))

    print()
    print("Focused Crawling is starting...")
    print()

    batches = []
    harvest_rates = []
    crawled_pages = []     
    rewards = []
    history_urls = {}       
    errors = 0
    total_per_period = 0
    time_start = time.time()
    tree_leafs = []
    tree_sizes = []
    while(True):
        print()
        if VERBOSE and agent.env.crawler_sys.times_verbose % VERBOSE_PERIOD == 0:
            print(f"{UNDERLINE}Timestep: {agent.env.current_step}{ENDC}")  
            print("Frontier's size:", agent.env.tree_frontier.root.frontier_size)
            print("Frontier's leafs:", len(agent.env.tree_frontier.leafs))
            print("Closure size:", len(agent.env.closure.closure))

        # Pop a webpage from tree frontier  
        page = agent.tree_policy(policy=POLICY)
        while page is None:
            agent.refreshFrontierLeafs()
            page = agent.tree_policy(policy=POLICY)
        print(f"Page fetched: {page.url}")

        # Perform a step in environment
        state_page, reward, done, _ = agent.env.step(action=page.id)
        if state_page == False:
            continue

        if VERBOSE and agent.env.crawler_sys.times_verbose % VERBOSE_PERIOD == 0:
            print(f"{OKBLUE}Different relevant domains: {len(agent.env.different_domains)}{ENDC}")
            print(page.x, page.relevant_parents, page.irrelevant_parents)
            print("Q-value:", page.qvalue)
            print(f"{OKGREEN}Reward: {reward}{ENDC}")
        
        # Store reward and harvest rate
        rewards.append(reward)
        harvest_rates.append(agent.env.harvestRate())
        
        # Save crawled page
        crawled_pages.append(page.url) 
        history_urls[page.url] = 1
        print(f"History URLs size: {len(history_urls)}")

        # Check for termination
        if done: 
            print('\n', "Crawling has been finished")
            break
        
        total_per_period += reward
        if (agent.env.current_step - 1) % TARGET_UPDATE_PERIOD == 0 and agent.env.current_step > 1:
            if LR_DECAY:
                # Learning Rate decay
                agent.decreaseLR()
                print("Learning Rate:", agent.getLR())
            print('Mean of Rewards during this period:', total_per_period / TARGET_UPDATE_PERIOD)
            total_per_period = 0
            gc.collect()

        if (agent.env.current_step - 1) % TREE_REFRESH_PERIOD == 0:
            # Refresh Frontier Leafs
            agent.refreshFrontierLeafs()
        
        try:
            # Extract outlinks of crawled page
            extractedURLS = agent.env.extractStateActions()     # list of Webpage
        except: 
            print("Exception CUDA extractStateActions")
            extractedURLS = []

        # Store record to experience replay buffer
        record = (page.x, page.id, reward)
        if extractedURLS != []:
            agent.buffer.insert( record )
        agent.env.tree_frontier.addSample(record, flag="exp")

        # Check for Target Q-Network update
        if agent.check_for_target_update():
            print("Target update")
            agent.updateTarget()

        # Q-Network Training
        try:
            if agent.env.current_step % REPLAY_PERIOD == 0:
                for t in range(TAKE_BATCHES):
                    agent.train()   
        except: 
            print("Exception CUDA train")
            save_file = "./saves/"
            extractedURLS = []

        # Update tree frontier
        agent.evaluate_and_updateFrontier(extractedURLS)

        tree_leafs.append(len(agent.env.tree_frontier.leafs))
        tree_sizes.append(agent.env.tree_frontier.root.frontier_size)

        if agent.env.current_step % 1000 == 0 and agent.env.current_step > 0 :
            # Save (harvest_rates, rewards, crawled_pages (urls), batches, TUP, errors)
            history = (harvest_rates, rewards, crawled_pages, tree_leafs, tree_sizes, len(agent.env.different_domains))
            history_path = f'{folder}{domain}_crawl_history_{machine}.pickle'
            with open(history_path, 'wb') as handle:
                pickle.dump(history, handle)

        # Harvest Rate result for this timestep
        if VERBOSE and agent.env.crawler_sys.times_verbose % VERBOSE_PERIOD == 0:
            print("Harvest Rate:", agent.env.harvestRate(), '\n')
        agent.env.crawler_sys.times_verbose += 1

        # tf reset default graph    
        tf.compat.v1.reset_default_graph()

    # Save (harvest_rates, rewards, crawled_pages (urls), batches, TUP, errors)
    history = (harvest_rates, rewards, crawled_pages, tree_leafs, tree_sizes, len(agent.env.different_domains))
    history_path = f'{folder}{domain}_crawl_history_{machine}.pickle'
    with open(history_path, 'wb') as handle:
        pickle.dump(history, handle) 
    print("Len history:", len(history[0]))

    if POLICY != "random":
        agent.q_network.model.save(f"DDQN_{domain}_{machine}")

    time_end = time.time()
    print("Crawling time:", (time_end - time_start) / 3600, "hours")


