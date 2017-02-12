import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.StringTokenizer;
import java.util.function.Predicate;

public class WhoToFollow {

    /*
    ** First Round Mapper - Indexing
    */
    public static class IndexingMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        public void map(Object key, Text values, Context context) throws IOException, InterruptedException {
            StringTokenizer st = new StringTokenizer(values.toString());

            IntWritable k = new IntWritable(Integer.parseInt(st.nextToken()));
            IntWritable friend1 = new IntWritable();
            while(st.hasMoreTokens()) {
                Integer friend = Integer.parseInt(st.nextToken());
                friend1.set(friend);
                context.write(friend1, k);
                friend1.set(-friend);
                context.write(k, friend1);
            }
        }
    }

    /*
    ** First Round Reducer - Indexing
    */

    public static class IndexingReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

        // The reduce method
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

            StringBuilder sb = new StringBuilder();
            while(values.iterator().hasNext()) {
                sb.append(" " + values.iterator().next().toString());
            }
            context.write(key, new Text(sb.toString()));


        }
    }

    /*
    ** Second Round Mapper - Similarity
    */
    public static class SimilarityMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        public void map(Object key, Text values, Context context) throws IOException, InterruptedException {
            // Key is ignored as it only stores the offset of the line in the text file
            StringTokenizer st = new StringTokenizer(values.toString());
            // seenFriends will store the friends we've already seen as we walk through the list of friends
            ArrayList<Integer> seenFriends = new ArrayList<>();
            // friend1 and friend2 will be the elements in the emitted pairs.
            IntWritable friend1 = new IntWritable();
            IntWritable friend2 = new IntWritable();
            String k = st.nextToken(); // discards first token (key)
            while (st.hasMoreTokens()) {
                Integer i = Integer.parseInt(st.nextToken());
                // if token < 0 => save the origin value
                if(i > 0) {
                    // For every friend Fi found in the values,
                    // we emit (Fi,Fj) and (Fj,Fi) for every Fj in the
                    // friends we have seen before. You can convince yourself
                    // that this will emit all (Fi,Fj) pairs for i!=j.
                    friend1.set(i);
                    for (Integer seenFriend : seenFriends) {
                        friend2.set(seenFriend);
                        context.write(friend1, friend2);
                        context.write(friend2, friend1);
                    }
                    seenFriends.add(friend1.get());
                } else {
                    context.write(new IntWritable(Integer.parseInt(k)), new IntWritable(i));
                }

            }
        }
    }

    /*
    ** Second Round Reducer - Similarity
    */
    public static class SimilarityReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

        // A private class to describe a recommendation.
        // A recommendation has a friend id and a number of friends in common.
        private static class Recommendation {

            // Attributes
            private int friendId;
            private int nCommonFriends;

            // Constructor
            public Recommendation(int friendId) {
                this.friendId = friendId;
                // A recommendation must have at least 1 common friend
                this.nCommonFriends = 1;
            }

            // Getters
            public int getFriendId() {
                return friendId;
            }

            public int getNCommonFriends() {
                return nCommonFriends;
            }

            // Other methods
            // Increments the number of common friends
            public void addCommonFriend() {
                nCommonFriends++;
            }

            // String representation used in the reduce output
            public String toString() {
                return friendId + "(" + nCommonFriends + ")";
            }

            // Finds a representation in an array
            public static Recommendation find(int friendId, ArrayList<Recommendation> recommendations) {
                for (Recommendation p : recommendations) {
                    if (p.getFriendId() == friendId) {
                        return p;
                    }
                }
                // Recommendation was not found!
                return null;
            }
        }

        // The reduce method
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            IntWritable user = key;
            // 'existingFriends' will store the friends of user 'user'
            // (the negative values in 'values').
            ArrayList<Integer> existingFriends = new ArrayList();
            // 'recommendedUsers' will store the list of user ids recommended
            // to user 'user'
            ArrayList<Integer> recommendedUsers = new ArrayList<>();
            while (values.iterator().hasNext()) {
                int value = values.iterator().next().get();
                if (value > 0) {
                    recommendedUsers.add(value);
                } else {
                    existingFriends.add(value);
                }
            }
            // 'recommendedUsers' now contains all the positive values in 'values'.
            // We need to remove from it every value -x where x is in existingFriends.
            // See javadoc on Predicate: https://docs.oracle.com/javase/8/docs/api/java/util/function/Predicate.html
            for (Integer friend : existingFriends) {
                recommendedUsers.removeIf(new Predicate<Integer>() {
                    @Override
                    public boolean test(Integer t) {
                        return t.intValue() == -friend.intValue();
                    }
                });
            }
            ArrayList<Recommendation> recommendations = new ArrayList<>();
            // Builds the recommendation array
            for (Integer userId : recommendedUsers) {
                Recommendation p = Recommendation.find(userId, recommendations);
                if (p == null) {
                    recommendations.add(new Recommendation(userId));
                } else {
                    p.addCommonFriend();
                }
            }
            // Sorts the recommendation array
            // See javadoc on Comparator at https://docs.oracle.com/javase/8/docs/api/java/util/Comparator.html
            recommendations.sort(new Comparator<Recommendation>() {
                @Override
                public int compare(Recommendation t, Recommendation t1) {
                    return -Integer.compare(t.getNCommonFriends(), t1.getNCommonFriends());
                }
            });
            // Builds the output string that will be emitted
            StringBuffer sb = new StringBuffer(""); // Using a StringBuffer is more efficient than concatenating strings
            for (int i = 0; i < recommendations.size() && i < 10; i++) {
                Recommendation p = recommendations.get(i);
                sb.append(p.toString() + " ");
            }
            Text result = new Text(sb.toString());
            context.write(user, result);
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        Configuration conf2 = new Configuration();


        Job job = Job.getInstance(conf, "who to follow - 1");
        job.setJarByClass(WhoToFollow.class);

        Job job2 = Job.getInstance(conf2, "who to follow - 2");
        job2.setJarByClass(WhoToFollow.class);


        job.setMapperClass(IndexingMapper.class);
        job.setReducerClass(IndexingReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("output"));
        job.waitForCompletion(true);

        job2.setMapperClass(SimilarityMapper.class);
        job2.setReducerClass(SimilarityReducer.class);
        job2.setOutputKeyClass(IntWritable.class);
        job2.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job2, new Path("output"));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));
        job2.waitForCompletion(true);
    }
}
