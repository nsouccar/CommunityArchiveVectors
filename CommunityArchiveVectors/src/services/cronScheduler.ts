import cron from 'node-cron';
import { EmbeddingSync } from './embeddingSync';

export class CronScheduler {
  private syncer: EmbeddingSync;

  constructor() {
    this.syncer = new EmbeddingSync();
  }

  /**
   * Start the hourly sync job
   */
  startHourlySync() {
    console.log('⏰ Starting hourly sync scheduler...');
    console.log('   Will check for new tweets every hour on the hour\n');

    // Run every hour at minute 0
    // Format: "minute hour * * *"
    cron.schedule('0 * * * *', async () => {
      console.log(`\n⏰ Hourly sync triggered at ${new Date().toISOString()}`);

      try {
        await this.syncer.syncTweets('2025-10-01T00:00:00');
      } catch (error) {
        console.error('❌ Hourly sync failed:', error);
      }
    });

    console.log('✅ Scheduler started!');
    console.log('   Next sync will run at the top of the next hour\n');
  }

  /**
   * Run initial sync immediately, then start scheduler
   */
  async startWithInitialSync() {
    console.log('🚀 Running initial sync...\n');

    try {
      await this.syncer.syncTweets('2025-10-01T00:00:00');
    } catch (error) {
      console.error('❌ Initial sync failed:', error);
    }

    this.startHourlySync();
  }
}

// For testing - run if called directly
if (require.main === module) {
  const scheduler = new CronScheduler();
  scheduler.startWithInitialSync();
}
