import csv
import random
import secrets
from datetime import datetime, timedelta

# Configuration
NUM_WALLETS = 5000
NUM_CLEAN_TXNS = 25000
DAYS_SPAN = 90
OUTPUT_TXNS = 'transactions.csv'
OUTPUT_LABELS = 'labels.csv'

# Set seeds for reproducibility
random.seed(42)

def generate_wallet_address():
    """Generates a random Ethereum-style address."""
    return "0x" + secrets.token_hex(20)

def generate_timestamp(start_date, end_date):
    """Generates a random timestamp between two dates."""
    delta = end_date - start_date
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start_date + timedelta(seconds=random_second)

def main():
    print("Initializing...")
    
    # 1. Generate Wallet IDs
    wallets = [generate_wallet_address() for _ in range(NUM_WALLETS)]
    # Map wallet to an index for easier role assignment tracking/verification if needed
    wallet_map = {w: i for i, w in enumerate(wallets)}
    
    # Track labels: Default 0 (Clean)
    # 1 = Illicit (Criminal, Mule, Boss, Cyclic attributes)
    labels = {w: 0 for w in wallets}
    
    transactions = []
    
    base_time = datetime.now()
    start_time = base_time - timedelta(days=DAYS_SPAN)
    
    print(f"Generating {NUM_CLEAN_TXNS} clean transactions...")
    
    # 2. Clean Transactions
    for _ in range(NUM_CLEAN_TXNS):
        src, tgt = random.sample(wallets, 2)
        amount = random.uniform(0.5, 50.0)
        timestamp = generate_timestamp(start_time, base_time)
        
        # Token Type: 80% ETH, 20% USDT
        token_type = 'ETH' if random.random() < 0.8 else 'USDT'
        
        transactions.append({
            'Source': src,
            'Target': tgt,
            'Amount': round(amount, 6),
            'Timestamp': timestamp,
            'Token_Type': token_type
        })

    print("Generating Smurfing patterns...")
    
    # 3. Smurfing Pattern
    # Select 10 Criminals
    # Filter out wallets already used if STRICT separation is needed, 
    # but in reality criminals also do clean txns. We just flag the ID.
    available_wallets = list(wallets)
    random.shuffle(available_wallets)
    
    criminals = available_wallets[:10]
    boss = available_wallets[10] # Single Boss for all, or could be one per criminal. Prompt implies 'a Boss wallet'.
    mules_pool = available_wallets[11:11+500] # 50 mules per criminal * 10 criminals = 500 mules
    
    # Mark labels
    for w in criminals: labels[w] = 1
    labels[boss] = 1
    for w in mules_pool: labels[w] = 1
    
    for i, crim in enumerate(criminals):
        # Assign 50 specific mules to this criminal
        my_mules = mules_pool[i*50 : (i+1)*50]
        
        # Split 2000 ETH into 50 txns -> 40 ETH each
        amount_per_mule = 2000.0 / 50.0 # 40.0
        
        # Placement Time
        placement_time = generate_timestamp(start_time, base_time - timedelta(hours=2)) # Ensure space for forwarding
        
        for mule in my_mules:
            # Step 1: Criminal -> Mule
            transactions.append({
                'Source': crim,
                'Target': mule,
                'Amount': round(amount_per_mule, 6),
                'Timestamp': placement_time,
                'Token_Type': 'ETH' # Usually base currency for fees
            })
            
            # Step 2: Mule -> Boss (Peeling)
            # Deduct 1-3% fee
            fee_pct = random.uniform(0.01, 0.03)
            forward_amount = amount_per_mule * (1.0 - fee_pct)
            
            # Delay 15-90 mins
            delay_mins = random.randint(15, 90)
            forward_time = placement_time + timedelta(minutes=delay_mins)
            
            transactions.append({
                'Source': mule,
                'Target': boss,
                'Amount': round(forward_amount, 6),
                'Timestamp': forward_time,
                'Token_Type': 'ETH'
            })

    print("Generating Cyclic patterns...")
    
    # 4. Cyclic Pattern
    # 5 Loops of 5: A->B->C->D->E->A
    # Use new wallets to avoid overlap mess, or reuse check? 
    # Let's pick from remaining wallets to keep roles clear for the label file.
    used_count = 11 + 500
    cyclic_candidates = available_wallets[used_count : used_count + 25]
    
    for i in range(5):
        loop_wallets = cyclic_candidates[i*5 : (i+1)*5]
        # Mark labels
        for w in loop_wallets: labels[w] = 1
        
        # Flow A->B->C->D->E->A
        # Random start amount? Not specified, picking reasonable range e.g. 100-500
        current_amount = random.uniform(100, 500)
        loop_start_time = generate_timestamp(start_time, base_time - timedelta(days=1))
        
        # The loop (Nodes 0->1, 1->2, 2->3, 3->4, 4->0)
        for j in range(5):
            src = loop_wallets[j]
            tgt = loop_wallets[(j+1) % 5]
            
            transactions.append({
                'Source': src,
                'Target': tgt,
                'Amount': round(current_amount, 6),
                'Timestamp': loop_start_time + timedelta(minutes=30*j), # Staggered
                'Token_Type': 'ETH'
            })
            
            # Decrease amount by 0.5% for next hop
            current_amount = current_amount * 0.995

    # 5. Exports
    print("Sort and export...")
    
    # Sort transactions by timestamp for realism
    transactions.sort(key=lambda x: x['Timestamp'])
    
    # Write Transactions
    with open(OUTPUT_TXNS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Source', 'Target', 'Amount', 'Timestamp', 'Token_Type'])
        writer.writeheader()
        writer.writerows(transactions)
        
    print(f"Wrote {len(transactions)} transactions to {OUTPUT_TXNS}")
        
    # Write Labels
    # ID, Label
    with open(OUTPUT_LABELS, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Wallet_ID', 'Label'])
        for w in wallets:
            writer.writerow([w, labels[w]])
            
    print(f"Wrote {len(labels)} labels to {OUTPUT_LABELS}")
    print("Done.")

if __name__ == "__main__":
    main()
