<div align="center">
  <div>&nbsp;</div>

## SEID QUERIES

Account Balance Query:
seid query bank balances sei1n7p4c4sjxap8nvhfwhgss6xyht2v60fc0423ey --count-total --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json

Query tx
seid query tx --type=hash 2FB3170B40CA7E6A0464F2AB92434754EE5F0841DAB3C6C481659DEA07734539 --chain-id pacific-1 --node https://sei-rpc.brocha.in/

Qeury events
seid query txs --events 'wasm.action=create_pair' --height 49599356 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --limit 1

seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward' --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json

Query token factory
seid query tokenfactory params --height 49741856 --chain-id pacific-1 --node https://sei-rpc.brocha.in/

</div>
